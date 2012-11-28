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
ADD_CUSTOM_COMMAND(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/operation-xml.h
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/operation.xml
    COMMENT "Generating XML header with operations meta info" VERBATIM
    COMMAND
        echo
            "namespace ranally {\\n"
            "static char const* const operations_xml = \"\\"
                > ${CMAKE_CURRENT_BINARY_DIR}/operation-xml.h
    COMMAND
        cat ${CMAKE_CURRENT_SOURCE_DIR}/operation.xml
            # Escape quotes.
            | sed "s/\\\"/\\\\\\\"/g"
            # Escape new-lines .
            | sed "s/\$/ \\\\/"
              >> ${CMAKE_CURRENT_BINARY_DIR}/operation-xml.h
    COMMAND
        echo
            "\";"
            "}\\n"
                >> ${CMAKE_CURRENT_BINARY_DIR}/operation-xml.h
)
SET(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/operation-pskel.cxx
    operation
    operations
    parameter
    result
    operation_xml_parser
)
ADD_LIBRARY(ranally_operation
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_operation
    ranally_core
    ${EXPAT_LIBRARY}
)
ADD_CUSTOM_TARGET(operation-xml.h
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/operation-xml.h
)
