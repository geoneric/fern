copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/raster-1.asc
    DESTINATION_FILENAME raster-1.asc
    DESTINATION_FILE_PATHNAMES_LIST outputs)


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


copy_test_file(
    SOURCE_FILE_PATHNAME ${CMAKE_CURRENT_BINARY_DIR}/constant-1.h5
    DESTINATION_FILENAME write_only.h5
    PERMISSIONS WRITE_ONLY
    DESTINATION_FILE_PATHNAMES_LIST outputs)


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
