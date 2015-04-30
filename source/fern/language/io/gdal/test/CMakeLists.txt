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
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/raster-2.asc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${FERN_DATA_DIR}/test/raster-2.asc .
    DEPENDS ${FERN_DATA_DIR}/test/raster-2.asc
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/raster-2.asc
)


add_custom_target(
    language_io_gdal_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    gdal_dataset_test
)
add_unit_tests(
    SCOPE language_io_gdal
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_io_gdal
    DEPENDENCIES language_io_gdal_test_data)
