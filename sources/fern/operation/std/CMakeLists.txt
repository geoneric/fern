ADD_SUBDIRECTORY(test)

ADD_CUSTOM_COMMAND(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/operation-xml.h
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/operation.xml
    COMMENT "Generating XML header with operations meta info" VERBATIM
    COMMAND
        echo
            "namespace fern {\\n"
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
    abs
    add
    int32
    operations
    read
    write
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_operation_std ${SOURCES})
TARGET_LINK_LIBRARIES(fern_operation_std
    fern_operation_core
    fern_io
)
ADD_CUSTOM_TARGET(operation-xml.h
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/operation-xml.h
)
