set(TEST_NAMES
    expression_type_test
    expression_types_test
)
add_unit_tests(
    SCOPE language_operation_core
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_operation_core)
