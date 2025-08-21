"""
MongoDB client for EagleRouter with Atlas Vector Search support.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from common import DEFAULT_ELO_SCORE, MODEL_MATCH_PROMPT_EMBEDDING_INDEX

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB client for managing training samples and model matrices."""

    def __init__(self, connection_string: str, database_name: str = "eagle_router"):
        """
        Initialize MongoDB client.

        Args:
            connection_string: MongoDB Atlas connection string
            database_name: Name of the database to use
        """
        try:
            self.client = MongoClient(connection_string)
            # Test connection
            self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

        self.db = self.client[database_name]

        # Initialize collections
        self.matches = self.db["matches"]
        self.elo_scores = self.db["elo_scores"]
        self.models = self.db["models"]

        # Create indexes
        self._setup_indexes()

    def _setup_indexes(self):
        """Set up necessary indexes for efficient querying."""
        try:
            self.matches.create_index([("prompt", 1)], unique=True)

            logger.info("Indexes created successfully")
        except OperationFailure as e:
            logger.warning(f"Some indexes may already exist: {e}")

    def add_new_match(
        self,
        prompt: str,
        embedding: np.ndarray,
        model_a: str,
        model_b: str,
        score: int,
        all_models: List[str],
    ):
        """
        Add or update a training sample with match result matrix.

        Args:
            prompt: The input prompt
            embedding: The embedding vector for the prompt
            model_a: Name of the first model
            model_b: Name of the second model
            score: 0 if model_a wins, 1 if model_b wins, 2 for draw
            all_models: List of all registered models

        Returns:
            Inserted/updated document ID
        """
        # Check if this prompt already exists
        existing = self.matches.find_one({"prompt": prompt})

        if existing:
            # Update existing prompt's match matrix
            match_matrix = np.array(existing.get("match_result_matrix", []))
            models = existing.get("models", [])

            # Expand matrix if needed for new models
            if set(all_models) != set(models):
                match_matrix, models = self._expand_prompt_matrix(
                    match_matrix, models, all_models
                )

            # Update the specific match result
            idx_a = models.index(model_a)
            idx_b = models.index(model_b)
            match_matrix[idx_a, idx_b] = score

            # Update document
            self.matches.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "match_result_matrix": match_matrix.tolist(),
                        "models": models,
                        "embedding": embedding.tolist()
                        if isinstance(embedding, np.ndarray)
                        else embedding,
                    }
                },
            )
        else:
            # Create new prompt with match matrix
            n = len(all_models)
            match_matrix = np.full((n, n), -1, dtype=int)
            np.fill_diagonal(match_matrix, -1)

            # Set the match result
            idx_a = all_models.index(model_a)
            idx_b = all_models.index(model_b)
            match_matrix[idx_a, idx_b] = score

            document = {
                "prompt": prompt,
                "embedding": embedding.tolist()
                if isinstance(embedding, np.ndarray)
                else embedding,
                "match_result_matrix": match_matrix.tolist(),
                "models": all_models,
            }

            self.matches.insert_one(document)

    def _expand_prompt_matrix(
        self, matrix: np.ndarray, existing_models: List[str], all_models: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Expand a prompt's match result matrix to include new models.

        Args:
            matrix: Existing match result matrix
            existing_models: List of models in the existing matrix
            all_models: List of all models including new ones

        Returns:
            Tuple of (expanded_matrix, updated_models_list)
        """
        # Find new models
        new_models = [m for m in all_models if m not in existing_models]
        if not new_models:
            return matrix, existing_models

        # Calculate new size
        n_existing = len(existing_models)
        n_new = len(new_models)
        n_total = n_existing + n_new

        # Create new expanded matrix
        new_matrix = np.full((n_total, n_total), -1, dtype=int)

        # Copy existing values
        if n_existing > 0 and matrix.size > 0:
            new_matrix[:n_existing, :n_existing] = matrix

        # Update models list
        updated_models = existing_models + new_models

        return new_matrix, updated_models

    def get_nearest_neighbors(
        self,
        query_embedding: np.ndarray,
        n_neighbors: int = 20,
        filter_models: Optional[Tuple[str, str]] = None,
    ) -> List[Dict]:
        """
        Find nearest neighbors using Atlas Vector Search.

        Args:
            query_embedding: Query embedding vector
            n_neighbors: Number of neighbors to retrieve
            filter_models: Optional tuple of (model_a, model_b) to filter results

        Returns:
            List of nearest neighbor documents
        """
        # Convert numpy array to list for MongoDB
        embedding_list = (
            query_embedding.tolist()
            if isinstance(query_embedding, np.ndarray)
            else query_embedding
        )

        # Build the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": MODEL_MATCH_PROMPT_EMBEDDING_INDEX,
                    "path": "embedding",
                    "queryVector": embedding_list,
                    "numCandidates": n_neighbors,
                    "limit": n_neighbors,
                }
            }
        ]

        # Add filter if specific models are requested
        if filter_models:
            model_a, model_b = filter_models
            pipeline.append({"$match": {"models": {"$all": [model_a, model_b]}}})

        # Add score field for vector search similarity
        pipeline.append(
            {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}}
        )

        try:
            results = list(self.matches.aggregate(pipeline))
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # Fallback to regular search without vector similarity
            if filter_models:
                model_a, model_b = filter_models
                return list(
                    self.matches.find(
                        {
                            "$or": [
                                {"model_a": model_a, "model_b": model_b},
                                {"model_a": model_b, "model_b": model_a},
                            ]
                        }
                    ).limit(n_neighbors)
                )
            return []

    def get_elo_scores(
        self, models: List[str], p: float, n: int, k: int
    ) -> Dict[str, float]:
        """
        Get or create ELO scores for given models.

        Args:
            models: List of model names

        Returns:
            Dictionary mapping model names to their ELO scores
        """
        # Try to get existing scores
        elo_doc = self.elo_scores.find_one({"p": p, "n": n, "k": k})

        if elo_doc:
            existing_scores = elo_doc.get("scores", {})

            # Initialize any new models with default score
            for model in models:
                if model not in existing_scores:
                    existing_scores[model] = DEFAULT_ELO_SCORE

            # Save if we added new models
            if set(models) - set(elo_doc.get("scores", {}).keys()):
                self.save_elo_scores(existing_scores, p, n, k)

            return existing_scores
        else:
            # Create new scores dictionary with default ELO starting score
            elo_scores = {model: DEFAULT_ELO_SCORE for model in models}

            # update elo score with pre-trained results if available
            import json

            with open("./notebooks/trained_elo_scores.json", "r") as f:
                scores_per_p_n_k = json.load(f)
            for p_n_k_dict in scores_per_p_n_k:
                if (
                    p_n_k_dict["p"] == p
                    and p_n_k_dict["n"] == n
                    and p_n_k_dict["k"] == k
                ):
                    for model, score in p_n_k_dict["scores"].items():
                        if model in elo_scores:
                            elo_scores[model] = score

            # Save new scores
            self.save_elo_scores(elo_scores, p, n, k)
            return elo_scores

    def save_elo_scores(self, elo_scores: Dict[str, float], p: float, n: int, k: int):
        """
        Save ELO scores to database.

        Args:
            elo_scores: Dictionary mapping model names to their ELO scores
            p: Global/local weighting parameter
            n: Number of neighbors
            k: ELO sensitivity parameter
        """
        # Ensure no legacy or unrelated fields (like 'type') are present
        doc = {
            "scores": elo_scores,
            "models": list(elo_scores.keys()),
            "p": p,
            "n": n,
            "k": k,
        }

        # Try to update existing document
        result = self.elo_scores.update_one(
            {"p": p, "n": n, "k": k},
            {"$set": doc},
            upsert=False,
        )
        # If no document was matched, upsert (replace or insert) to avoid duplicate key errors
        if result.matched_count == 0:
            self.elo_scores.insert_one(doc)

    def register_model(self, model_name: str, metadata: Optional[Dict] = None):
        """
        Register a new model in the database.

        Args:
            model_name: Name of the model
            metadata: Optional metadata about the model
        """
        document = {"name": model_name, "metadata": metadata or {}}

        self.models.update_one({"name": model_name}, {"$set": document}, upsert=True)

    def get_all_models(self) -> List[str]:
        """
        Get list of all registered models.

        Returns:
            List of model names
        """
        return [doc["name"] for doc in self.models.find({}, {"name": 1})]

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
