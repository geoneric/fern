set(TEST_NAMES
    collect_no_data_indices_test
    discard_domain_errors_test
    mark_no_data_by_value_test
    parallel_execution_policy_test
)
add_unit_tests(
    SCOPE algorithm_policy
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_algorithm_policy_objects>
    LINK_LIBRARIES
        fern_core
)
