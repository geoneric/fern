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
add_library_and_object_library(fern_language_ast_visitor ${SOURCES})
target_link_libraries(fern_language_ast_visitor
    fern_language_ast_core
)
