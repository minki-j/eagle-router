"""
Improved EagleRouter with MongoDB integration and matrix-based scoring system.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from mongodb_client import MongoDBClient
from common import DEFAULT_ELO_SCORE

logger = logging.getLogger(__name__)


class EagleRouter:
    """
    Eagle Router for intelligent model selection using matrix-based scoring.

    This router uses a combination of global ELO scores (from all training data)
    and local scores (from nearest neighbors) to select the best model for a query.
    """

    def __init__(
        self,
        mongodb_uri: str,
        P: float,
        N: int,
        K: int,
        database_name: str = "eagle_router",
    ):
        """
        Initialize the Eagle Router.

        Args:
            mongodb_uri: MongoDB Atlas connection string
            database_name: Name of the database to use
            P: Weight for global vs local scores (0-1)
            N: Number of nearest neighbors for local scoring
            K: ELO sensitivity parameter
        """
        self.P = P
        self.N = N
        self.K = K
        # Initialize MongoDB client
        self.db = MongoDBClient(mongodb_uri, database_name)

        # Cache for current ELO scores and models
        self.elo_scores = None
        self.models = None
        self.model_cost_map = {
            "gpt-4o-mini-2024-07-18": 0.75,
            "gpt-4o-2024-08-06": 12.5,
        }

        # Load existing matrices if available
        self._load_matrices()

    def _load_matrices(self):
        """Load existing ELO scores from database."""
        models = self.db.get_all_models()
        if models:
            self.models = models

            self.elo_scores = self.db.get_elo_scores(
                models, p=self.P, n=self.N, k=self.K
            )
            logger.info(f"Loaded ELO scores for {len(models)} models")

    def add_model(self, model_name: str, metadata: Optional[Dict] = None):
        """
        Add a new model to the router.

        Args:
            model_name: Name of the model to add
            metadata: Optional metadata about the model
        """
        # Register in database
        self.db.register_model(model_name, metadata)

        # Update local model list
        if self.models is None:
            self.models = [model_name]
        elif model_name not in self.models:
            self.models.append(model_name)

        # Update ELO scores
        self.elo_scores = self.db.get_elo_scores(
            self.models, p=self.P, n=self.N, k=self.K
        )

        logger.info(f"Added model: {model_name}")

    def add_new_match(
        self, prompt: str, embedding: np.ndarray, model_a: str, model_b: str, score: int
    ):
        """
        Add a training sample and update global matrices.

        Args:
            prompt: The input prompt
            embedding: The embedding vector for the prompt
            model_a: Name of the first model
            model_b: Name of the second model
            score: 0 if model_a wins, 1 if model_b wins, 2 for draw

        Returns:
            ID of the inserted training sample
        """
        # Ensure models are registered
        if model_a not in (self.models or []):
            self.add_model(model_a)
        if model_b not in (self.models or []):
            self.add_model(model_b)

        # Add training sample to database with match result matrix
        if self.models is None:
            self.models = []
        self.db.add_new_match(prompt, embedding, model_a, model_b, score, self.models)

        # Update ELO scores
        self._update_elo_scores(model_a, model_b, score)

        # Persist ELO scores to database
        if self.elo_scores is not None:
            self.db.save_elo_scores(self.elo_scores, p=self.P, n=self.N, k=self.K)

        logger.info(f"Added training sample: {model_a} vs {model_b}, score={score}")

    def _calculate_s_value_for_tied_match(self, model_a: str, model_b: str) -> float:
        """
        Calculate the S value (expected score) for a tied match between two models,
        taking into account their relative costs.

        The S value is used to adjust ELO updates for draws, rewarding the less expensive model:
            - If both models have the same cost, returns 0.5 (true tie).
            - If model_a is more expensive than model_b, returns a value < 0.5,
              penalizing model_a for higher cost.
            - If model_b is more expensive than model_a, returns a value > 0.5,
              rewarding model_a for being less expensive.
        The adjustment is proportional to the cost difference, normalized by the maximum
        cost difference among all models.

        Args:
            model_a: Name of the first model.
            model_b: Name of the second model.

        Returns:
            A float representing the adjusted S value for a tied match, in the range [0, 0.7].
        """
        global_max_cost = max(self.model_cost_map.values())
        golbal_min_cost = min(self.model_cost_map.values())
        max_cost_diff = abs(global_max_cost - golbal_min_cost)

        cost_diff = self.model_cost_map[model_a] - self.model_cost_map[model_b]
        return 0.5 - 0.2 * (cost_diff / max_cost_diff)

    def _update_elo_scores(self, model_a: str, model_b: str, score: int):
        """
        Update ELO scores based on a match result.

        Args:
            model_a: Name of the first model
            model_b: Name of the second model
            score: 0 if model_a wins, 1 if model_b wins, 2 for draw
        """
        if self.elo_scores is None:
            raise ValueError("ELO scores not initialized")

        # Convert score to ELO format (0->1, 1->0, 2->0.5)
        if score == 0:  # model_a wins
            S = 1.0
        elif score == 1:  # model_b wins
            S = 0.0
        else:  # draw
            S = self._calculate_s_value_for_tied_match(model_a, model_b)

        # Get current ELO scores
        elo_a = self.elo_scores.get(model_a)
        elo_b = self.elo_scores.get(model_b)

        if elo_a is None or elo_b is None:
            raise ValueError("ELO scores not initialized")

        # Calculate new ELO scores
        new_elo_a, new_elo_b = self._calculate_new_elo(elo_a, elo_b, S)

        # Update ELO scores dictionary
        self.elo_scores[model_a] = new_elo_a
        self.elo_scores[model_b] = new_elo_b

    def _calculate_new_elo(
        self, R_a: float, R_b: float, S: float
    ) -> Tuple[float, float]:
        """
        Update ELO scores based on match result.

        Args:
            R_a: Current ELO score of model a
            R_b: Current ELO score of model b
            S: 1 if a wins, 0 if b wins, 0.5 for draw

        Returns:
            Tuple of (new_elo_a, new_elo_b)
        """
        E_a = 1 / (1 + 10 ** ((R_b - R_a) / 400))
        R_a_new = R_a + self.K * (S - E_a)
        R_b_new = R_b + self.K * ((1 - S) - (1 - E_a))
        return R_a_new, R_b_new

    def _get_model_elo(self, model: str) -> float:
        """
        Get current ELO score for a model.

        Args:
            model: The model to get the score for

        Returns:
            Current ELO score (defaults to 1500 if no data)
        """
        if self.elo_scores is None or model not in self.elo_scores:
            raise ValueError("ELO scores not initialized")

        return self.elo_scores[model]

    def _get_local_global_combined_score(
        self, query_embedding: np.ndarray, model_a: str, model_b: str
    ) -> Dict[str, float]:
        # Calculate global scores
        global_score_a = self._get_model_elo(model_a)
        global_score_b = self._get_model_elo(model_b)
        logger.info(f"<global> a: {global_score_a:.2f}, b: {global_score_b:.2f}")

        # Calculate local scores from nearest neighbors
        result = self._calculate_local_scores(query_embedding, model_a, model_b)
        if result is None:
            logger.warning("No local scores found. Using global scores only.")
            return {model_a: global_score_a, model_b: global_score_b}

        local_score_a, local_score_b = result
        logger.info(f"<local> a: {local_score_a:.2f}, b: {local_score_b:.2f}")

        # Combine scores
        combined_a = self.P * global_score_a + (1 - self.P) * local_score_a
        combined_b = self.P * global_score_b + (1 - self.P) * local_score_b
        logger.info(f"<combined> a: {combined_a:.2f}, b: {combined_b:.2f}")

        return {model_a: combined_a, model_b: combined_b}

    def route(
        self, query_embedding: np.ndarray, candidate_models: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Route a query to the best model based on global and local scores.

        Args:
            query_embedding: Embedding vector for the query
            candidate_models: Optional list of models to consider (defaults to all)

        Returns:
            Dictionary mapping model names to their combined scores
        """
        if self.models is None or len(self.models) == 0:
            raise ValueError("No models registered. Add models first.")

        # Use all models if none specified
        if candidate_models is None:
            candidate_models = self.models
        else:
            # Validate candidate models
            for model in candidate_models:
                if model not in self.models:
                    raise ValueError(f"Model {model} not registered")

        # If only one model, return it
        if len(candidate_models) == 1:
            return {candidate_models[0]: 100.0}

        # For now, we'll handle pairwise comparison for 2 models
        # This can be extended to multi-model comparison later
        if len(candidate_models) == 2:
            model_a, model_b = candidate_models
            return self._get_local_global_combined_score(
                query_embedding, model_a, model_b
            )
        else:
            # For multiple models, compute global and local scores for each
            model_scores = {}

            for model in candidate_models:
                # Get global ELO score
                global_score = self._get_model_elo(model)

                # Calculate local score against all other candidates
                local_score = self._calculate_multi_model_local_score(
                    query_embedding, model, candidate_models
                )

                # Combine scores
                combined_score = self.P * global_score + (1 - self.P) * local_score
                model_scores[model] = combined_score

            return model_scores

    def _calculate_local_scores(
        self, query_embedding: np.ndarray, model_a: str, model_b: str
    ) -> Tuple[float, float] | None:
        """
        Calculate local scores for two models based on nearest neighbors.

        This uses the match result matrices from similar prompts to determine local performance.

        Args:
            query_embedding: Query embedding vector
            model_a: First model name
            model_b: Second model name

        Returns:
            Tuple of (local_score_a, local_score_b)
        """
        # Initialize with default ELO
        #! Using default score instead of global score to make local more local.
        local_elo_a = DEFAULT_ELO_SCORE
        local_elo_b = DEFAULT_ELO_SCORE

        # Get nearest neighbors - these contain match result matrices for similar prompts
        neighbors = self.db.get_nearest_neighbors(
            query_embedding,
            self.N,
            filter_models=(model_a, model_b),
        )
        if len(neighbors) == 0:
            logger.warning("No neighbors found for query embedding")
            return None

        # Calculate local ELO based on match results from similar prompts
        for neighbor in neighbors:
            match_matrix = neighbor.get("match_result_matrix")
            models = neighbor.get("models")

            if match_matrix is None or models is None:
                # Fallback to old format if exists
                n_model_a = neighbor.get("model_a")
                n_model_b = neighbor.get("model_b")
                n_score = neighbor.get("score")

                if n_score is not None and n_model_a and n_model_b:
                    # Handle old format
                    if n_model_a == model_a and n_model_b == model_b:
                        S = 1.0 if n_score == 0 else (0.0 if n_score == 1 else 0.5)
                    elif n_model_a == model_b and n_model_b == model_a:
                        S = 0.0 if n_score == 0 else (1.0 if n_score == 1 else 0.5)
                    else:
                        continue
                    local_elo_a, local_elo_b = self._calculate_new_elo(
                        local_elo_a, local_elo_b, S
                    )
                continue

            # Use match result matrix
            if model_a in models and model_b in models:
                idx_a = models.index(model_a)
                idx_b = models.index(model_b)

                match_result = match_matrix[idx_a][idx_b]

                if match_result != -1:  # Has data
                    # match_result: 0 = row wins (model_a), 1 = col wins (model_b), 2 = draw
                    if match_result == 0:  # model_a wins
                        S = 1.0
                    elif match_result == 1:  # model_b wins
                        S = 0.0
                    else:  # draw
                        S = self._calculate_s_value_for_tied_match(model_a, model_b)

                    # Update local ELO based on this similar prompt's result
                    local_elo_a, local_elo_b = self._calculate_new_elo(
                        local_elo_a, local_elo_b, S
                    )

        return local_elo_a, local_elo_b

    def _calculate_multi_model_local_score(
        self,
        query_embedding: np.ndarray,
        target_model: str,
        candidate_models: List[str],
    ) -> float:
        """
        Calculate local score for a model against multiple candidates.

        Args:
            query_embedding: Query embedding vector
            target_model: Model to calculate score for
            candidate_models: List of all candidate models

        Returns:
            Local score for the target model
        """
        # Get nearest neighbors
        neighbors = self.db.get_nearest_neighbors(query_embedding, self.N)

        # Count wins/losses for target model
        wins = 0
        losses = 0
        draws = 0
        total = 0

        for neighbor in neighbors:
            n_model_a = neighbor.get("model_a")
            n_model_b = neighbor.get("model_b")
            n_score = neighbor.get("score")

            if n_score is None:
                continue

            # Check if target model is involved
            if n_model_a == target_model:
                if n_model_b in candidate_models:
                    total += 1
                    if n_score == 0:  # target wins
                        wins += 1
                    elif n_score == 1:  # target loses
                        losses += 1
                    else:  # draw
                        draws += 1
            elif n_model_b == target_model:
                if n_model_a in candidate_models:
                    total += 1
                    if n_score == 1:  # target wins
                        wins += 1
                    elif n_score == 0:  # target loses
                        losses += 1
                    else:  # draw
                        draws += 1

        # Calculate win rate based score
        if total > 0:
            win_rate = (wins + 0.5 * draws) / total
            # Convert to ELO-like score (1500 = baseline, higher is better)
            return DEFAULT_ELO_SCORE + 400 * (win_rate - 0.5)

        return DEFAULT_ELO_SCORE

    def get_best_model(
        self,
        query_embedding: np.ndarray,
        candidate_models: Optional[List[str]] = None,
    ) -> Union[str, Dict[str, float]]:
        """
        Get the best model for a query.

        Args:
            query_embedding: Embedding vector for the query
            candidate_models: Optional list of models to consider

        Returns:
            Name of the best model
        """
        scores = self.route(query_embedding, candidate_models)
        if not scores:
            raise ValueError("No scores available")

        return max(scores.items(), key=lambda x: x[1])[0]

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()
