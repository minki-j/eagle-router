import faiss
import numpy as np
import pandas as pd


class DB:
    def __init__(self, df_path="data/train_1000.parquet"):
        self.df = pd.read_parquet(df_path)

        embeddings = np.vstack(self.df["prompt_embedding"].values).astype("float32")

        # Create FAISS index for L2 distance
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        # clean up memory
        del embeddings
        self.df.drop(columns=["prompt_embedding"], inplace=True)

    def get_total_rows(self):
        return len(self.df)

    def get_winner(self, row_indice, model_a, model_b):
        # check if indice is valid
        if row_indice < 0 or row_indice >= self.get_total_rows():
            raise ValueError(f"Row indice {row_indice} is not valid")

        if (
            self.df.iloc[row_indice][model_a + "/score"]
            > self.df.iloc[row_indice][model_b + "/score"]
        ):
            return model_a
        elif (
            self.df.iloc[row_indice][model_a + "/score"]
            < self.df.iloc[row_indice][model_b + "/score"]
        ):
            return model_b
        else:
            return None

    def get_nearest_neighbors(self, query_prompt_embedding, N):
        """ """
        # Ensure prompt_embedding is the right shape and type
        if query_prompt_embedding.ndim == 1:
            query_prompt_embedding = query_prompt_embedding.reshape(1, -1)
        query_prompt_embedding = query_prompt_embedding.astype("float32")

        _, indices = self.index.search(query_prompt_embedding, N)
        rows_of_nearest_neighbor = self.df.iloc[indices[0]]

        # remove embedding column
        rows_of_nearest_neighbor = rows_of_nearest_neighbor.drop(
            columns=["prompt", "question", "gold_answer"]
        )

        return rows_of_nearest_neighbor
