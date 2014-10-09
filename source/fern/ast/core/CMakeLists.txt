ADD_SUBDIRECTORY(test)

SET(SOURCES
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
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_ast_core ${SOURCES})
TARGET_LINK_LIBRARIES(fern_ast_core
    fern_operation_core
    fern_core
)
