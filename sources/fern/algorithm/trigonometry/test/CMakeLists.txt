SET(TEST_NAMES
    atan_test
    cos_test
    sin_test
    tan_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_trigonometry_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_trigonometry_${NAME}
        fern_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(
        NAME algorithm_trigonometry_${NAME}
        COMMAND algorithm_trigonometry_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
