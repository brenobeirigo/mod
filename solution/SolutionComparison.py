from collections import defaultdict
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from mod.util.file_util import read_json_file
from solution.data_util import movingaverage
from .Solution import Solution


class SolutionComparison:
    colors_default = [
        "k",
        "g",
        "r",
        "b",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
    ]

    def __init__(self, filepath):
        self.data = read_json_file(filepath)
        self.show_labels_on_right = self.data["show_labels_on_right"]
        self.output_path = self.data["output"]
        self.create_dirs_recursevely()
        self.show_shadow = self.data["show_shadow"]
        self.category_dict = self.data["category_comparison"]
        self.x_label = self.data["x_label"]
        self.mark_every = self.data["mark_every"]
        self.window = self.data["window"]
        self.id = self.data["id"]
        self.sources = self.data["source_list"]
        self.colors = self.data["color_list"]
        self.iteration_limit = self.data["iteration_limit"]
        self.linewidths = self.data["linewidth_list"]
        self.markers = self.data.get("marker_list", [None] * len(self.sources))
        self.labels = self.data["label_list"]
        self.context = self.data["context"]
        self.dpi = self.data["dpi"]
        self.figure_format = self.data["figure_format"]
        self.yticks = dict()
        self.yticks_labels = dict()
        self.category_solutions = defaultdict(list)
        self.solution_stats_dict = dict()
        self.load_solutions()

    def create_dirs_recursevely(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def load_solutions(self):

        for source, source_label in self.get_source_label_pair_list():

            solution = Solution(source, source_label)

            for category in self.category_dict:
                self.compute_category_solution_stats(category, solution)
                self.configure_categoty_ticks(category)
                self.category_solutions[category].append(solution)

    def get_source_label_pair_list(self):
        return zip(self.sources, self.labels)

    def configure_categoty_ticks(self, category):
        self.yticks[category] = np.linspace(**self.category_dict[category]["ticks"])
        ticks_format = self.category_dict[category]['ticks_format']
        self.yticks_labels[category] = [f"{p:{ticks_format}}" for p in self.yticks[category]]

    def compute_category_solution_stats(self, category, solution):
        category_values = solution.get_values_from_category(category)
        category_values = self.get_sublist_until_max_interations(category_values)
        self.solution_stats_dict[SolutionComparison.get_standard_label(category)] = category_values

    def get_sublist_until_max_interations(self, category_values):
        return category_values[:self.iteration_limit]

    @staticmethod
    def get_standard_label(label):
        return label.lower().replace("-", "").replace(" ", "_")

    def save_outcome(self):
        df_outcome = pd.DataFrame(self.solution_stats_dict)
        df_outcome = df_outcome[sorted(df_outcome.columns.values)]
        df_outcome.to_csv("outcome_tuning.csv", index=False)

    def plot_comparison(self):

        sns.set_context(self.context)
        np.set_printoptions(precision=3)

        fig, axs = plt.subplots(1, len(self.category_dict), figsize=(8 * len(self.category_dict), 6))

        for plot_index, category in enumerate(self.category_dict):

            for solution_index, solution in enumerate(self.category_solutions[category]):

                solution_values = solution.get_values_from_category(category)

                if self.show_shadow:
                    axs[plot_index].plot(
                        solution_values,
                        color=self.colors[solution_index],
                        linewidth=self.linewidths[solution_index],
                        marker=self.markers[solution_index],
                        alpha=0.25,
                        label="",
                    )

                mavg = movingaverage(solution_values, self.window)

                axs[plot_index].plot(
                    mavg,
                    color=self.colors[solution_index],
                    linewidth=self.linewidths[solution_index],
                    marker=self.markers[solution_index],
                    fillstyle="none",
                    markevery=self.mark_every,
                    label=solution.label,
                )

                # And add a special annotation for the group we are interested in
                if self.show_labels_on_right:
                    axs[plot_index].text(self.iteration_limit+10, mavg[-1], solution.label, horizontalalignment='left', size='small', color='k')

                # axs[i].set_title(vst)
                axs[plot_index].set_xlabel(self.x_label)
                axs[plot_index].set_ylabel(category)
                axs[plot_index].set_xlim(0, len(solution_values))
                axs[plot_index].set_yticks(self.yticks[category])
                axs[plot_index].set_yticklabels(self.yticks_labels[category])

        legend_pos = dict()
        legend_pos["policy"] = "center right"

        plt.legend(
            loc=legend_pos.get(self.id, "lower right"),
            frameon=False,
            bbox_to_anchor=(1, 0, 0, 1),  # (0.5, -0.15),
            ncol=1,
            # title="Max. #cars/location"
        )

        # plt.show()
        print(f'Saving "{self.id}.{self.figure_format}"...')
        plt.savefig(f"{self.output_path}/{self.id}.{self.figure_format}", bbox_inches="tight", dpi=self.dpi)

    def __str__(self) -> str:
        return super().__str__()
