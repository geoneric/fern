set(TEST_NAMES
    array_test
    masked_array_test
    masked_raster_test
    masked_scalar_test
    raster_test
)
add_unit_tests(
    SCOPE feature_core
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_feature_core_objects>
)
