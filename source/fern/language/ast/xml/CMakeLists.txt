add_test_conditionally(test)

add_parser_generation_command(syntax_tree
    --namespace-map http://www.geoneric.eu/fern=fern::language)

set(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/syntax_tree-pskel.cxx
    xml_parser
)
add_object_library(
    TARGET fern_language_ast_xml_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
