add_test_conditionally(test)

set(SOURCES
    assignment_vertex
    ast_vertex
    attribute_vertex
    expression_vertex
    function_call_vertex
    function_definition_vertex
    if_vertex
    module_vertex
    name_vertex
    number_vertex
    operation_vertex
    operator_vertex
    return_vertex
    scope_vertex
    sentinel_vertex
    statement_vertex
    string_vertex
    subscript_vertex
    while_vertex
)
add_library_and_object_library(fern_language_ast_core ${SOURCES})
target_link_libraries(fern_language_ast_core
    fern_language_operation_core
    fern_core
)
