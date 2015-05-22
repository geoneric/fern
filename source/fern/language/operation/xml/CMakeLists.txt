add_test_conditionally(test)

add_parser_generation_command(operation
    --namespace-map http://www.geoneric.eu/fern=fern::language)

set(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/operation-pskel.cxx
    operation_xml_parser
)
add_library(fern_language_operation_xml_objects OBJECT ${SOURCES})
