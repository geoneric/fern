add_unit_tests(
    SCOPE algorithm_statistic
    NAMES
        binary_min_test
        binary_max_test
        count_test
        mean_test
        unary_min_test
        unary_max_test
        sum_test
    LINK_LIBRARIES
        fern::algorithm
        fern::feature
)
