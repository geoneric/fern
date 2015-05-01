set(TEST_NAMES
    operation_xml_parser_test
)
add_unit_tests(
    SCOPE language_operation_xml
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_operation_xml)
