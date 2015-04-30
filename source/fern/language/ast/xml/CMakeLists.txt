add_test_conditionally(test)

add_parser_generation_command(syntax_tree
    --namespace-map http://www.geoneric.eu/fern=fern::language)

set(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree-pskel.cxx
    xml_parser
)
add_library_and_object_library(fern_language_ast_xml ${SOURCES})
target_link_libraries(fern_language_ast_xml
    fern_language_ast_core
    ${EXPAT_LIBRARY}
)
