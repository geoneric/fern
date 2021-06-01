add_unit_tests(
    SCOPE algorithm_algebra_vector
    NAMES
        gradient_test.cc
        laplacian_test.cc
        lax_test.cc
    LINK_LIBRARIES
        fern::algorithm
        fern::feature
)
