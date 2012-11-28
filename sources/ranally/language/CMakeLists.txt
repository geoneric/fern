ADD_SUBDIRECTORY(test)

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/syntax_tree.map.in
    ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree.map
)
ADD_CUSTOM_COMMAND(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/ranally-pskel.hxx
        ${CMAKE_CURRENT_BINARY_DIR}/ranally-pskel.cxx
    COMMAND
        ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
            --type-map ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree.map
            --namespace-map http://www.jemig.eu/ranally=ranally
            ${CMAKE_CURRENT_SOURCE_DIR}/ranally.xsd
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree.map
        ${CMAKE_CURRENT_SOURCE_DIR}/ranally.xsd
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating XML parser ѕkeleton for scripts" VERBATIM
)
SET(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/ranally-pskel.cxx
    algebra_parser
    annotate_visitor
    assignment_vertex
    ast_dot_visitor
    copy_visitor
    dot_visitor
    execute_visitor
    expression_vertex
    function_vertex
    identify_visitor
    if_vertex
    flowgraph_dot_visitor
    name_vertex
    number_vertex
    operation_vertex
    operator_vertex
    optimize_visitor
    script_vertex
    script_visitor
    statement_vertex
    string_vertex
    symbol_table
    syntax_vertex
    thread_visitor
    validate_visitor
    visitor
    while_vertex
    xml_parser
)
ADD_LIBRARY(ranally_language
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_language
    ranally_operation
    ranally_python
    ranally_core
    ${EXPAT_LIBRARY}
    ${PYTHON_LIBRARIES}
)
