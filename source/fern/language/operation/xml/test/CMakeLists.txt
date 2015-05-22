set(TEST_NAMES
    operation_xml_parser_test
)
add_unit_tests(
    SCOPE language_operation_xml
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_language_feature_visitor_objects>
        $<TARGET_OBJECTS:fern_language_operation_core_objects>
        $<TARGET_OBJECTS:fern_language_operation_xml_objects>
    LINK_LIBRARIES
        fern_core
        ${EXPAT_LIBRARY}
)
