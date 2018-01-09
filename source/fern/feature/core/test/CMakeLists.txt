add_unit_tests(
    SCOPE feature_core
    NAMES
        array_test
        masked_array_test
        masked_raster_test
        masked_scalar_test
        raster_test
    LINK_LIBRARIES
        fern::feature
)
