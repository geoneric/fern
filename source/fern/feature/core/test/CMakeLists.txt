add_unit_tests(
    SCOPE feature_core
    NAMES
        array_test.cc
        masked_array_test.cc
        masked_raster_test.cc
        masked_scalar_test.cc
        raster_test.cc
    LINK_LIBRARIES
        fern::feature
)
