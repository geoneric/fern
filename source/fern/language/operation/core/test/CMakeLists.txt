set(TEST_NAMES
    expression_type_test
    expression_types_test
)
add_unit_tests(
    SCOPE language_operation_core
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_language_operation_core_objects>
        $<TARGET_OBJECTS:fern_language_feature_visitor_objects>
    LINK_LIBRARIES
        fern_core
)
