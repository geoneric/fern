SET(TEST_NAMES
    absolute_test
    add_test
    divide_test
    equal_test
    greater_than_test
    less_than_test
    multiply_test
    sqrt_test
    subtract_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_algebra_elementary_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_algebra_elementary_${NAME}
        fern_core
        fern_algorithm_core
        fern_feature_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(
        NAME algorithm_algebra_elementary_${NAME}
        COMMAND algorithm_algebra_elementary_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
