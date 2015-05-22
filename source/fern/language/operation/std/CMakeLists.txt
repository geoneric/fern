add_test_conditionally(test)

add_custom_command(
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
set(SOURCES
    abs
    add
    int32
    operations
    read
    write
)
add_object_library(
    TARGET fern_language_operation_std_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
add_custom_target(operation-xml.h
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/operation-xml.h
)
