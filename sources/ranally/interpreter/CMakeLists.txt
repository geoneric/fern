ADD_SUBDIRECTORY(test)

SET(SOURCES
    attribute_value
    execute_visitor
    interpreter
    value
)
ADD_LIBRARY(ranally_interpreter
    ${SOURCES}
)
ADD_DEPENDENCIES(ranally_interpreter
    # This header is generated in the operation project.
    operation-xml.h
)
TARGET_LINK_LIBRARIES(ranally_interpreter
    ranally_feature
    ranally_ast_visitor
    ranally_ast_xml
    ranally_operation_xml
    ranally_script
)
