set(TEST_NAMES
    attribute_type_visitor_test
)
add_unit_tests(
    SCOPE language_feature_visitor
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_language_feature_visitor_objects>
    LINK_LIBRARIES
        fern_core
)
