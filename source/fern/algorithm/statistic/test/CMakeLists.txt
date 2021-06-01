add_unit_tests(
    SCOPE algorithm_statistic
    NAMES
        binary_min_test.cc
        binary_max_test.cc
        count_test.cc
        mean_test.cc
        unary_min_test.cc
        unary_max_test.cc
        sum_test.cc
    LINK_LIBRARIES
        fern::algorithm
        fern::feature
)
