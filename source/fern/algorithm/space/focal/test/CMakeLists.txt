set(TEST_NAMES
    slope_test
)
add_unit_tests(
    SCOPE algorithm_spatial_focal
    NAMES ${TEST_NAME}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_algorithm_policy_objects>
        $<TARGET_OBJECTS:fern_algorithm_core_objects>
    LINK_LIBRARIES
        fern_feature
)
