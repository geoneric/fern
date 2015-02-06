set(TEST_NAMES
    collect_no_data_indices_test
    discard_domain_errors_test
    mark_no_data_by_value_test
    parallel_execution_policy_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_policy
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy)
endforeach()
