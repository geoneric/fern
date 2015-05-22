set(TEST_NAMES
    abs_test
    operations_test
    print_test
)
add_unit_tests(
    SCOPE language_operation_std
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language
        fern_core
)
