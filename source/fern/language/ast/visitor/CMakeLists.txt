add_test_conditionally(test)

set(SOURCES
    annotate_visitor
    ast_dot_visitor
    ast_visitor
    copy_visitor
    dot_visitor
    identify_visitor
    io_visitor
    flowgraph_dot_visitor
    module_visitor
    optimize_visitor
    thread_visitor
    validate_visitor
)
add_object_library(
    TARGET fern_language_ast_visitor_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
