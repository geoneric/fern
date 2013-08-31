ADD_SUBDIRECTORY(test)

# CONFIGURE_FILE(
#     ${CMAKE_CURRENT_SOURCE_DIR}/syntax_tree.map.in
#     ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree.map
# )
# ADD_CUSTOM_COMMAND(
#     OUTPUT
#         ${CMAKE_CURRENT_BINARY_DIR}/ranally-pskel.hxx
#         ${CMAKE_CURRENT_BINARY_DIR}/ranally-pskel.cxx
#     COMMAND
#         ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
#             --type-map ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree.map
#             --namespace-map http://www.jemig.eu/ranally=ranally
#             ${CMAKE_CURRENT_SOURCE_DIR}/ranally.xsd
#     DEPENDS
#         ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree.map
#         ${CMAKE_CURRENT_SOURCE_DIR}/ranally.xsd
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMENT "Generating XML parser ѕkeleton for scripts" VERBATIM
# )

ADD_PARSER_GENERATION_COMMAND(syntax_tree
    --namespace-map http://www.jemig.eu/ranally=ranally)

SET(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree-pskel.cxx
    xml_parser
)
ADD_LIBRARY(ranally_ast_xml
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_ast_xml
    ranally_ast_core
    ${EXPAT_LIBRARY}
)
