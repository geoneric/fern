add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/raster-1.asc
    COMMAND ${CMAKE_COMMAND} -E copy
        ${FERN_DATA_DIR}/test/raster-1.asc
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${FERN_DATA_DIR}/test/raster-1.asc
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/raster-1.asc
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/constant-1.h5
    COMMAND $<TARGET_FILE:fern> execute
        ${FERN_DATA_DIR}/test/constant-1.frn
    DEPENDS fern ${FERN_DATA_DIR}/test/constant-1.frn
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/constant-1.h5
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/write_only.h5
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/constant-1.h5
        ${CMAKE_CURRENT_BINARY_DIR}/write_only.h5
    COMMAND chmod 222 ${CMAKE_CURRENT_BINARY_DIR}/write_only.h5
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/constant-1.h5
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/write_only.h5
)


add_custom_target(
    language_io_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    drivers_test
)
add_unit_tests(
    SCOPE language_io
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_io
    DEPENDENCIES language_io_test_data)
