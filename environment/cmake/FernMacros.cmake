MACRO(ADD_PARSER_GENERATION_COMMAND
        BASENAME)
    CONFIGURE_FILE(
        ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.map.in
        ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.map
    )

    ADD_CUSTOM_COMMAND(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}-pskel.hxx
            ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}-pskel.cxx
        COMMAND
            ${XSD_EXECUTABLE} cxx-parser --xml-parser expat
                --type-map ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.map
                ${ARGN}
                ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.xsd
        DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.map
            ${CMAKE_CURRENT_SOURCE_DIR}/${BASENAME}.xsd
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
ENDMACRO()


# Create a static library and an object library.
# BASENAME: Name of static library to create. The object library will be 
#           named ${BASENAME}_objects.
# SOURCES : Sources that are part of the libraries. Any argument that comes
#           after the BASENAME is treated as a source.
MACRO(ADD_LIBRARY_AND_OBJECT_LIBRARY
        BASENAME)
    SET(SOURCES ${ARGN})
    ADD_LIBRARY(${BASENAME}
        ${SOURCES}
    )
    ADD_LIBRARY(${BASENAME}_objects OBJECT
        ${SOURCES}
    )
ENDMACRO()
