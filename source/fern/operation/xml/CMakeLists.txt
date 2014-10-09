ADD_SUBDIRECTORY(test)

ADD_PARSER_GENERATION_COMMAND(operation
    --namespace-map http://www.geoneric.eu/fern=fern)

SET(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/operation-pskel.cxx
    operation_xml_parser
)
ADD_LIBRARY(fern_operation_xml
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(fern_operation_xml
    fern_operation_core
    # fern_operation_std
    ${EXPAT_LIBRARY}
)
