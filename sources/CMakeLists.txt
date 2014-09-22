INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

### # TODO By default, this check should be disabled.
### # Verify that headers are self-sufficient. This is especially important
### # for headers that are part of the public API.
### GET_DIRECTORY_PROPERTY(INCLUDES INCLUDE_DIRECTORIES)
### FILE(GLOB_RECURSE ALGORITHM_HEADER_PATHNAMES "fern/algorithm/*.h")
### FILE(GLOB_RECURSE CORE_HEADER_PATHNAMES "fern/core/*.h")
### FILE(GLOB_RECURSE FEATURE_HEADER_PATHNAMES "fern/feature/*.h")
### SET(PUBLIC_HEADER_PATHNAMES
###     # Order matters. First test the most basic headers and after that the
###     # ones that use those headers.
###     ${CORE_HEADER_PATHNAMES}
###     ${FEATURE_HEADER_PATHNAMES}
###     ${ALGORITHM_HEADER_PATHNAMES}
### )
### VERIFY_HEADERS_ARE_SELF_SUFFICIENT(
###     OFFSET_PATHNAME
###         ${CMAKE_CURRENT_SOURCE_DIR}/
###     INCLUDES
###         ${INCLUDES}
###     LIBRARIES
###         ${FERN_EXTERNAL_LIBRARIES}
###         ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
###     FLAGS
###         ${CMAKE_CXX_FLAGS}
###     HEADER_PATHNAMES
###         ${PUBLIC_HEADER_PATHNAMES}
### )

IF(DOXYGEN_FOUND)
    # SET(FERN_DOXYGEN_INPUT
    #     ${CMAKE_CURRENT_SOURCE_DIR}/fern
    #     ${CMAKE_CURRENT_BINARY_DIR}/fern
    #     # ${XSD_INCLUDE_DIRS}/xsd/cxx
    # )

    LIST(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/fern.md)
    IF(FERN_FERN_ALGORITHM_REQUIRED)
        LIST(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/algorithm)
    ENDIF()
    IF(FERN_FERN_CORE_REQUIRED)
        LIST(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/core)
    ENDIF()
    IF(FERN_FERN_EXAMPLE_REQUIRED)
        LIST(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/example)
    ENDIF()
    IF(FERN_FERN_FEATURE_REQUIRED)
        LIST(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/feature)
    ENDIF()
    # TODO Add more...

    STRING(REPLACE ";" " " FERN_DOXYGEN_INPUT "${_API_SOURCES}")

    INCLUDE(DoxygenDoc)

    INSTALL(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
        DESTINATION doc/api
    )
ENDIF()
ADD_SUBDIRECTORY(fern)
IF(FERN_REQUIRED_MALLOC_SYMBOLS_AVAILABLE)
    ADD_SUBDIRECTORY(tools)
ENDIF()
