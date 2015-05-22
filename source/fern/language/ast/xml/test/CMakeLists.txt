set(TEST_NAMES
    xml_parser_test
)
add_unit_tests(
    SCOPE language_ast_xml
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language
        fern_core
)
