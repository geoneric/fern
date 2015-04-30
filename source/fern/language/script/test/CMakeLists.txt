set(TEST_NAMES
    algebra_parser_test
)
add_unit_tests(
    SCOPE language_script
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_script)
