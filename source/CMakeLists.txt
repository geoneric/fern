include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_subdirectory(fern)
if(FERN_REQUIRED_MALLOC_SYMBOLS_AVAILABLE)
    add_subdirectory(tool)
endif()


### # TODO By default, this check should be disabled.
### # Verify that headers are self-sufficient. This is especially important
### # for headers that are part of the public API.
### get_directory_property(INCLUDES INCLUDE_DIRECTORIES)
### file(GLOB_RECURSE ALGORITHM_HEADER_PATHNAMES "fern/algorithm/*.h")
### file(GLOB_RECURSE CORE_HEADER_PATHNAMES "fern/core/*.h")
### file(GLOB_RECURSE FEATURE_HEADER_PATHNAMES "fern/feature/*.h")
### set(PUBLIC_HEADER_PATHNAMES
###     # Order matters. First test the most basic headers and after that the
###     # ones that use those headers.
###     ${CORE_HEADER_PATHNAMES}
###     ${FEATURE_HEADER_PATHNAMES}
###     ${ALGORITHM_HEADER_PATHNAMES}
### )
### verify_headers_are_self_sufficient(
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


if(FERN_BUILD_DOCUMENTATION)
    list(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/fern.md)
    list(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/group.dox)

    if(FERN_FERN_ALGORITHM_REQUIRED)
        list(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/algorithm)
    endif()
    if(FERN_FERN_CORE_REQUIRED)
        list(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/core)
    endif()
    if(FERN_FERN_EXAMPLE_REQUIRED)
        list(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/example)
    endif()
    if(FERN_FERN_FEATURE_REQUIRED)
        list(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/feature)
    endif()
    if(FERN_FERN_IO_REQUIRED)
        list(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/io)
        list(APPEND _API_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/fern/io)
    endif()
    if(FERN_BUILD_LANGUAGE)
        list(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/language)
    endif()
    if(FERN_FERN_PYTHON_REQUIRED)
        list(APPEND _API_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fern/python)
    endif()

    # TODO Add more...

    list(APPEND _API_SOURCES ${FERN_EXTERNAL_SOURCES})

    string(REPLACE ";" " " DEVBASE_DOXYGEN_INPUT "${_API_SOURCES}")

    include(DoxygenDoc)

    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
        DESTINATION doc/fern/api
    )
endif()
