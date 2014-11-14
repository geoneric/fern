SET(TEST_NAMES
    collect_no_data_indices_test
    discard_domain_errors_test
    mark_no_data_by_value_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(algorithm_policy ${NAME})
ENDFOREACH()
