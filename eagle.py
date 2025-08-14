from db import DB


class EagleRouter:
    def __init__(self, P=0.5, N=20, K=32):
        """
        P: weight for global vs local scores
        N: number of nearest neighbors
        K: ELO sensitivity
        """
        self.P, self.N, self.K = P, N, K

        # Mocking db with a temp class. Need to be optimized with actual db later.
        self.db = DB()

        # TODO: need to genearlize so that it can hangle more models.
        self.model_a = "gpt-4o-mini-2024-07-18"
        self.model_b = "gpt-4o-2024-08-06"

        # Initialize global scores
        self.global_scores = {
            self.model_a: 100,
            self.model_b: 100,
        }

    def _update_elo(self, R_a, R_b, S):
        # R_a: ELO score of model a
        # R_b: ELO score of model b
        # S: 1 if a wins, 0 if b wins, 0.5 for draw
        E_a = 1 / (1 + 10 ** ((R_b - R_a) / 400))
        R_a_new = R_a + self.K * (S - E_a)
        R_b_new = R_b + self.K * ((1 - S) - (1 - E_a))
        return R_a_new, R_b_new

    def _update_global(self, a, b, winner):
        if winner == a:
            S = 1
        elif winner == b:
            S = 0
        else:
            S = 0.5

        new_Ra, new_Rb = self._update_elo(
            self.global_scores[a], self.global_scores[b], S
        )

        self.global_scores[a], self.global_scores[b] = new_Ra, new_Rb

    def train_global_scores(self):
        total_training_rows = self.db.get_total_rows()
        for i in range(total_training_rows):
            winner = self.db.get_winner(i, self.model_a, self.model_b)
            self._update_global(self.model_a, self.model_b, winner)

    def route(self, query_prompt_embedding):
        local_scores = self.global_scores.copy()

        # Calculate local scores
        # First, get nearest neighbors from stored feedback
        rows_of_nearest_neighbor = self.db.get_nearest_neighbors(
            query_prompt_embedding, self.N
        )
        # For each row, update local scores
        for _, row in rows_of_nearest_neighbor.iterrows():
            if row[self.model_a + "/score"] > row[self.model_b + "/score"]:
                result = 1
            elif row[self.model_a + "/score"] < row[self.model_b + "/score"]:
                result = 0
            else:
                result = 0.5

            R_a_new, R_b_new = self._update_elo(
                local_scores[self.model_a], local_scores[self.model_b], result
            )
            local_scores[self.model_a], local_scores[self.model_b] = R_a_new, R_b_new

        # Combine local and global scores
        combined = {
            m: self.P * self.global_scores[m] + (1 - self.P) * local_scores[m]
            for m in [self.model_a, self.model_b]
        }

        # Return the best model
        return max(combined, key=combined.get)
