set(TEST_NAMES
    execute_visitor_test
    interpreter_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(interpreter_${NAME} ${NAME})
    target_link_libraries(interpreter_${NAME}
        fern_interpreter
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME interpreter_${NAME} COMMAND interpreter_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
