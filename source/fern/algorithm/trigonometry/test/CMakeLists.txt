add_unit_tests(
    SCOPE algorithm_trigonometry
    NAMES
        acos_test
        asin_test
        atan_test
        cos_test
        sin_test
        tan_test
    LINK_LIBRARIES
        fern::algorithm
)
