add_test_conditionally(test)

add_parser_generation_command(uncertml2
    --namespace-map http://www.uncertml.org/2.0=uncertml2
    --generate-polymorphic)
    # TODO Suppress validation because of issues in uncertml xsd.
    # --suppress-validation)

set(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/uncertml2-pskel.cxx
    uncertml2_parser
)
add_object_library(
    TARGET fern_language_io_uncertml2_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
