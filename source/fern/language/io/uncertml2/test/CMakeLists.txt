set(TEST_NAMES
    parser_test
)
add_unit_tests(
    SCOPE language_io_uncertml2
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_io_uncertml2
        fern_core)
