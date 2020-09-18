from solution.SolutionComparison import SolutionComparison

if __name__ == '__main__':
    comp = SolutionComparison("data/category_comparison/input/discount_factor_comparison.json")
    comp.plot_comparison()
    comparison_distributions = SolutionComparison("C:/Users/LocalAdmin/OneDrive/leap_forward/phd_project/reb/code/mod/config/A3_A4_1st_class_distribution/category_comparison/input/compare_distributions_A3_A4.json")
    comparison_distributions.plot_comparison()