ADD_SUBDIRECTORY(test)

SET(SOURCES
    argument
    compile_visitor
    compiler
    module
)
ADD_LIBRARY(fern_compiler
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(fern_compiler
    # fern_ast_visitor
    # fern_ast_xml
    # fern_operation_std
    # fern_script
    # fern_core
)
