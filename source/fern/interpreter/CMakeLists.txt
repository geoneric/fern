ADD_SUBDIRECTORY(test)

SET(SOURCES
    data_source
    data_sync
    dataset_source
    dataset_sync
    execute_visitor
    interpreter
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_interpreter ${SOURCES})
TARGET_LINK_LIBRARIES(fern_interpreter
    fern_ast_visitor
    fern_ast_xml
    fern_operation_std
    fern_script
)
