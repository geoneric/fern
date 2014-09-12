SET(TEST_NAMES
    and_test
    defined_test
    not_test
    or_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_algebra_boolean_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_algebra_boolean_${NAME}
        stdc++
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(
        NAME algorithm_algebra_boolean_${NAME}
        COMMAND algorithm_algebra_boolean_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
