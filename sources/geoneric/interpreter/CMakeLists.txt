ADD_SUBDIRECTORY(test)

SET(SOURCES
    data_source
    dataset_source
    execute_visitor
    interpreter
)
ADD_LIBRARY(geoneric_interpreter
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_interpreter
    geoneric_ast_visitor
    geoneric_ast_xml
    geoneric_operation_std
    geoneric_script
)
