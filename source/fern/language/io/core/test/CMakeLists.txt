add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/raster-1.asc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${FERN_DATA_DIR}/test/raster-1.asc .
    DEPENDS ${FERN_DATA_DIR}/test/raster-1.asc
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/raster-1.asc
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/write_only.asc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove -f write_only.asc
    COMMAND ${CMAKE_COMMAND} -E copy ${FERN_DATA_DIR}/test/raster-1.asc
        write_only.asc
    COMMAND chmod 222 write_only.asc
    DEPENDS ${FERN_DATA_DIR}/test/raster-1.asc
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/write_only.asc
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/raster-1-link.asc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove -f raster-1-link.asc
    COMMAND ${CMAKE_COMMAND} -E create_symlink raster-1.asc raster-1-link.asc
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/raster-1.asc
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/raster-1-link.asc
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/raster-1-dangling_link.asc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E create_symlink does_not_exist.asc
        raster-1-dangling_link.asc
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/raster-1-dangling_link.asc
)


add_custom_target(
    language_io_core_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    file_test
)
add_unit_tests(
    SCOPE language_io_core
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_io_core
    DEPENDENCIES language_io_core_test_data)
