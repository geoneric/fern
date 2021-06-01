add_unit_tests(
    SCOPE algorithm_trigonometry
    NAMES
        acos_test.cc
        asin_test.cc
        atan_test.cc
        cos_test.cc
        sin_test.cc
        tan_test.cc
    LINK_LIBRARIES
        fern::algorithm
)
