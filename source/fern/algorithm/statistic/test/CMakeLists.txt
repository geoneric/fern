set(TEST_NAMES
    binary_min_test
    binary_max_test
    count_test
    mean_test
    unary_min_test
    unary_max_test
    sum_test
)
add_unit_tests(
    SCOPE algorithm_statistic
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_algorithm_policy_objects>
        $<TARGET_OBJECTS:fern_algorithm_core_objects>
    LINK_LIBRARIES
        fern_feature
)
