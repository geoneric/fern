ADD_SUBDIRECTORY(test)

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
# SET(SOURCES
#     # ...
# )
# ADD_LIBRARY(ranally_operation_std
#     ${SOURCES}
# )
# TARGET_LINK_LIBRARIES(ranally_operation_std
#     ranally_core
# )
ADD_CUSTOM_TARGET(operation-xml.h
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/operation-xml.h
)
