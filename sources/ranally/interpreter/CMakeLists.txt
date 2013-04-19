ADD_SUBDIRECTORY(test)

SET(SOURCES
    execute_visitor
    interpreter
)
ADD_LIBRARY(ranally_interpreter
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_interpreter
    ranally_feature
    ranally_ast_visitor
    ranally_ast_xml
    ranally_operation_std
    ranally_script
)
