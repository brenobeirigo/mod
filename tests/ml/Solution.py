import pandas as pd


class Solution:

    def get_df_from_source(self):
        try:
            print(f"Reading {self.source}...")
            df = pd.read_csv(self.source)
        except Exception as e:
            print(f"Cannot load file!Exception: \"{e}\"")
            raise Exception(e)
        return df

    def __init__(self, source, label, iteration_limit=500):
        self.source = source
        self.label = label
        self.cut_iterations = iteration_limit
        self.df = self.get_df_from_source()
        self.total_reward = self.df["Total reward"][:self.cut_iterations]
        self.service_rate = self.df["Service rate"][:self.cut_iterations]
        self.time = self.df["time"][:self.cut_iterations]

    def get_values_from_category(self, category):
        return self.df[category].values
