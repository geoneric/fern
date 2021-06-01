add_unit_tests(
    SCOPE algorithm_policy
    NAMES
        collect_no_data_indices_test.cc
        discard_domain_errors_test.cc
        mark_no_data_by_value_test.cc
        parallel_execution_policy_test.cc
    LINK_LIBRARIES
        fern::algorithm
)
