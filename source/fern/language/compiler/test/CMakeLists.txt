set(TEST_NAMES
    compiler_test
    module_test
    parse_command_line_test
)
add_unit_tests(
    SCOPE language_compiler
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_compiler)
