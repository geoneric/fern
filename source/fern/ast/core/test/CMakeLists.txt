set(TEST_NAMES
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

macro(ADD_UNIT_TEST NAME)
    add_executable(ast_core_${NAME} ${NAME})
    target_link_libraries(ast_core_${NAME}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME ast_core_${NAME} COMMAND ast_core_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
