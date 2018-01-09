add_unit_tests(
    SCOPE algorithm_algebra_vector
    NAMES
        gradient_test
        laplacian_test
        lax_test
    LINK_LIBRARIES
        fern::algorithm
        fern::feature
)
