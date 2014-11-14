SET(TEST_NAMES
    binary_min_test
    binary_max_test
    count_test
    unary_min_test
    unary_max_test
    sum_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(algorithm_statistic ${NAME}
        fern_core fern_algorithm_core fern_feature_core)
ENDFOREACH()
