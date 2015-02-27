if(FERN_TEST)
    add_subdirectory(test)
endif()

set(SOURCES
    data_source
    data_sync
    dataset_source
    dataset_sync
    execute_visitor
    interpreter
)
add_library_and_object_library(fern_interpreter ${SOURCES})
target_link_libraries(fern_interpreter
    fern_ast_visitor
    fern_ast_xml
    fern_operation_std
    fern_script
)
