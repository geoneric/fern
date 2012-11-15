INCLUDE(UseBoost)
INCLUDE(UseExpat)
SET(XSD_ROOT ${PCRTEAM_PLATFORM})
INCLUDE(UseXsd)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_BINARY_DIR}
)

ADD_SUBDIRECTORY(Tests)

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/Operation.map.in
    ${CMAKE_CURRENT_BINARY_DIR}/Operation.map
)
ADD_CUSTOM_COMMAND(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/Operation-pskel.hxx
        ${CMAKE_CURRENT_BINARY_DIR}/Operation-pskel.cxx
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/Operation.map
        ${CMAKE_CURRENT_SOURCE_DIR}/Operation.xsd
    COMMENT "Generating XML parser ѕkeleton for operations" VERBATIM
    COMMAND
        ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
            --type-map ${CMAKE_CURRENT_BINARY_DIR}/Operation.map
            --namespace-map http://www.jemig.eu/ranally=ranally
            ${CMAKE_CURRENT_SOURCE_DIR}/Operation.xsd
)
ADD_CUSTOM_COMMAND(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/Operation-xml.h
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/Operation.xml
    COMMENT "Generating XML header with operations meta info" VERBATIM
    COMMAND
        echo
            "namespace ranally {\\n"
            "static char const* const operationsXml = \"\\"
                > ${CMAKE_CURRENT_BINARY_DIR}/Operation-xml.h
    COMMAND
        cat ${CMAKE_CURRENT_SOURCE_DIR}/Operation.xml
            # Escape quotes.
            | sed "s/\\\"/\\\\\\\"/g"
            # Escape new-lines .
            | sed "s/\$/ \\\\/"
              >> ${CMAKE_CURRENT_BINARY_DIR}/Operation-xml.h
    COMMAND
        echo
            "\";"
            "}\\n"
                >> ${CMAKE_CURRENT_BINARY_DIR}/Operation-xml.h
)
SET(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/Operation-pskel.cxx
    Operation.cc
    Operations.cc
    Parameter.cc
    Result.cc
    OperationXmlParser.cc
)
ADD_LIBRARY(RanallyOperation
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(RanallyOperation
    RanallyUtil
    ${EXPAT_LIBRARIES}
)
ADD_CUSTOM_TARGET(Operation-xml.h
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/Operation-xml.h
)
