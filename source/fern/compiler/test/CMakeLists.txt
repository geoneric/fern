set(TEST_NAMES
    compiler_test
    module_test
    parse_command_line_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(compiler_${NAME} ${NAME})
    target_link_libraries(compiler_${NAME}
        fern_compiler
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
    )
    add_test(NAME compiler_${NAME} COMMAND compiler_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
