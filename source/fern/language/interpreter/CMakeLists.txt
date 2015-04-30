add_test_conditionally(test)

set(SOURCES
    data_source
    data_sync
    dataset_source
    dataset_sync
    execute_visitor
    interpreter
)
add_library_and_object_library(fern_language_interpreter ${SOURCES})
target_link_libraries(fern_language_interpreter
    fern_language_ast_visitor
    fern_language_ast_xml
    fern_language_operation_std
    fern_language_script
)
