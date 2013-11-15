ADD_SUBDIRECTORY(test)

SET(SOURCES
    argument
    compile_visitor
    compiler
    module
)
ADD_LIBRARY(geoneric_compiler
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_compiler
    # geoneric_ast_visitor
    # geoneric_ast_xml
    # geoneric_operation_std
    # geoneric_script
    # geoneric_core
)
