set(TEST_NAMES
    attribute_type_visitor_test
)
add_unit_tests(
    SCOPE language_feature_visitor
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_feature_visitor)
