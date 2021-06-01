add_unit_tests(
    SCOPE algorithm_spatial_focal
    NAMES
        slope_test.cc
    LINK_LIBRARIES
        fern::algorithm
        fern::feature
)
