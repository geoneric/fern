SET(TEST_NAMES
    execute_visitor_test
    interpreter_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(interpreter_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(interpreter_${NAME}
        geoneric_interpreter
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME interpreter_${NAME} COMMAND interpreter_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
