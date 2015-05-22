add_test_conditionally(test)

set(SOURCES
    data_source
    data_sync
    dataset_source
    dataset_sync
    execute_visitor
    interpreter
)
add_object_library(
    TARGET fern_language_interpreter_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
# target_link_libraries(fern_language_interpreter
#     fern_language_ast_visitor
#     fern_language_ast_xml
#     fern_language_operation_std
#     fern_language_script
# )
