add_unit_tests(
    SCOPE algorithm_algebra_boole
    NAMES
        and_test.cc
        defined_test.cc
        nor_test.cc
        not_test.cc
        or_test.cc
        xnor_test.cc
        xor_test.cc
    LINK_LIBRARIES
        fern::algorithm
)
