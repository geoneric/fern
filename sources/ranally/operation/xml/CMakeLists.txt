ADD_SUBDIRECTORY(test)

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/operation.map.in
    ${CMAKE_CURRENT_BINARY_DIR}/operation.map
)
ADD_CUSTOM_COMMAND(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/operation-pskel.hxx
        ${CMAKE_CURRENT_BINARY_DIR}/operation-pskel.cxx
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/operation.map
        ${CMAKE_CURRENT_SOURCE_DIR}/operation.xsd
    COMMENT "Generating XML parser ѕkeleton for operations" VERBATIM
    COMMAND
        ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
            --type-map ${CMAKE_CURRENT_BINARY_DIR}/operation.map
            --namespace-map http://www.jemig.eu/ranally=ranally
            ${CMAKE_CURRENT_SOURCE_DIR}/operation.xsd
)
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
