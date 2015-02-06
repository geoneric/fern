set(TEST_NAMES
    binary_min_test
    binary_max_test
    count_test
    unary_min_test
    unary_max_test
    sum_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_statistic
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy fern_algorithm_core fern_core
            fern_feature_core)
endforeach()
