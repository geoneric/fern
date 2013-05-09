ADD_SUBDIRECTORY(test)

ADD_PARSER_GENERATION_COMMAND(uncertml2
    --namespace-map http://www.uncertml.org/2.0=uncertml2
    --generate-polymorphic)
    # TODO Suppress validation because of issues in uncertml xsd.
    # --suppress-validation)

SET(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/uncertml2-pskel.cxx
    uncertml2_parser
)
ADD_LIBRARY(ranally_io_uncertml2
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_io_uncertml2
    ranally_uncertainty
    ${EXPAT_LIBRARY}
)
