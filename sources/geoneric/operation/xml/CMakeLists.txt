ADD_SUBDIRECTORY(test)

ADD_PARSER_GENERATION_COMMAND(operation
    --namespace-map http://www.geoneric.eu=geoneric)

SET(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/operation-pskel.cxx
    operation_xml_parser
)
ADD_LIBRARY(geoneric_operation_xml
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_operation_xml
    geoneric_operation_core
    # geoneric_operation_std
    ${EXPAT_LIBRARY}
)
