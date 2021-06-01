add_unit_tests(
    SCOPE algorithm_convolution
    NAMES
        convolve_test.cc
    LINK_LIBRARIES
        fern::algorithm
        fern::feature
)
