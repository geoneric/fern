add_unit_tests(
    SCOPE algorithm_policy
    NAMES
        collect_no_data_indices_test
        discard_domain_errors_test
        mark_no_data_by_value_test
        parallel_execution_policy_test
    LINK_LIBRARIES
        fern::algorithm
)
