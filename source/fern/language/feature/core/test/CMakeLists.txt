set(TEST_NAMES
    array_value_test
    constant_attribute_test
    feature_test
    masked_array_value_test
    spatial_attribute_test
    spatial_domain_test
)
add_unit_tests(
    SCOPE language_feature_core
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_language_feature_core_objects>
    LINK_LIBRARIES
        fern_feature
)
