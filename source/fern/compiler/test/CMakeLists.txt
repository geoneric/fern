SET(TEST_NAMES
    compiler_test
    module_test
    parse_command_line_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(compiler_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(compiler_${NAME}
        fern_compiler
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
    )
    ADD_TEST(NAME compiler_${NAME} COMMAND compiler_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
