ADD_SUBDIRECTORY(test)

ADD_PARSER_GENERATION_COMMAND(operation
    --namespace-map http://www.jemig.eu/ranally=ranally)

SET(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/operation-pskel.cxx
    operation_xml_parser
)
ADD_LIBRARY(ranally_operation_xml
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_operation_xml
    ranally_operation_core
    # ranally_operation_std
    ${EXPAT_LIBRARY}
)
