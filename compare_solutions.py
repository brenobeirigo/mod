from solution.SolutionComparison import SolutionComparison

if __name__ == '__main__':
    comp = SolutionComparison("d:/bb/mod/config/A3_A4_1st_class_distribution/category_comparison/input/compare_discount_factors.json")
    comp.plot_comparison()
    
    comp_a2_b8 = SolutionComparison("d:/bb/mod/config/A3_A4_1st_class_distribution/category_comparison/input/compare_discount_factors_A2_B8.json")
    comp_a2_b8.plot_comparison()

    comp_adp_tune_b = SolutionComparison("d:/bb/mod/config/adp_tune/category_comparison/input/adp_tune_B.json")
    comp_adp_tune_b.plot_comparison()
    comparison_distributions = SolutionComparison("d:/bb/mod/config/A3_A4_1st_class_distribution/category_comparison/input/compare_distributions_A3_A4.json")
    comparison_distributions.plot_comparison()