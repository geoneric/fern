SET(TEST_NAMES
    assignment_vertex_test
    ast_vertex_test
    expression_vertex_test
    function_call_vertex_test
    function_definition_vertex_test
    if_vertex_test
    module_vertex_test
    name_vertex_test
    number_vertex_test
    operator_vertex_test
    return_vertex_test
    statement_vertex_test
    string_vertex_test
    while_vertex_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(ast_core_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(ast_core_${NAME}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME ast_core_${NAME} COMMAND ast_core_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
