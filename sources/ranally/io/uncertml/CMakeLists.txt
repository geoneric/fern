ADD_SUBDIRECTORY(test)

ADD_PARSER_GENERATION_COMMAND(uncertml-2.0
    --namespace-map http://www.uncertml.org/2.0=uncertml
    --generate-polymorphic)

SET(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/uncertml-2.0-pskel.cxx
    # xml_parser
)
ADD_LIBRARY(ranally_io_uncertml_2.0
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_io_uncertml_2.0
    # ranally_ast_core
    ${EXPAT_LIBRARY}
)
