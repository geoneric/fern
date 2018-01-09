add_unit_tests(
    SCOPE algorithm_algebra_boole
    NAMES
        and_test
        defined_test
        nor_test
        not_test
        or_test
        xnor_test
        xor_test
    LINK_LIBRARIES
        fern::algorithm
)
