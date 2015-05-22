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
