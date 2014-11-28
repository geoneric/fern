SET(TEST_NAMES
    absolute_test
    add_test
    divide_test
    equal_test
    greater_equal_test
    greater_test
    less_test
    less_equal_test
    multiply_test
    pow_test
    sqrt_test
    subtract_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE algorithm_algebra_elementary
        NAME ${NAME}
        LINK_LIBRARIES fern_core fern_algorithm_core fern_feature_core)
ENDFOREACH()
