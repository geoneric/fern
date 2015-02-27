if(FERN_TEST)
    add_subdirectory(test)
endif()

add_parser_generation_command(uncertml2
    --namespace-map http://www.uncertml.org/2.0=uncertml2
    --generate-polymorphic)
    # TODO Suppress validation because of issues in uncertml xsd.
    # --suppress-validation)

set(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/uncertml2-pskel.cxx
    uncertml2_parser
)
add_library(fern_io_uncertml2
    ${SOURCES}
)
target_link_libraries(fern_io_uncertml2
    fern_uncertainty
    ${EXPAT_LIBRARY}
)
