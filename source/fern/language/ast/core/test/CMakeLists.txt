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
add_unit_tests(
    SCOPE language_ast_core
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES )
