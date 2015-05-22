copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/raster-1.asc
    DESTINATION_FILENAME raster-1.asc
    DESTINATION_FILE_PATHNAMES_LIST outputs)
copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/raster-1.asc
    DESTINATION_FILENAME write_only.asc
    PERMISSIONS WRITE_ONLY
    DESTINATION_FILE_PATHNAMES_LIST outputs)


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
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_language_io_core_objects>
    LINK_LIBRARIES
        fern_core
    DEPENDENCIES
        language_io_core_test_data
)
