add_subdirectory(test)

# configure_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/syntax_tree.map.in
#     ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree.map
# )
# add_custom_command(
#     OUTPUT
#         ${CMAKE_CURRENT_BINARY_DIR}/fern-pskel.hxx
#         ${CMAKE_CURRENT_BINARY_DIR}/fern-pskel.cxx
#     COMMAND
#         ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
#             --type-map ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree.map
#             --namespace-map http://www.jemig.eu/fern=fern
#             ${CMAKE_CURRENT_SOURCE_DIR}/fern.xsd
#     DEPENDS
#         ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree.map
#         ${CMAKE_CURRENT_SOURCE_DIR}/fern.xsd
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMENT "Generating XML parser ѕkeleton for scripts" VERBATIM
# )

add_parser_generation_command(syntax_tree
    --namespace-map http://www.geoneric.eu/fern=fern)

set(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree-pskel.cxx
    xml_parser
)
add_library_and_object_library(fern_ast_xml ${SOURCES})
target_link_libraries(fern_ast_xml
    fern_ast_core
    ${EXPAT_LIBRARY}
)
