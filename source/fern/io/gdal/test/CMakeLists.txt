add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/unreadable.tif
    COMMAND ${CMAKE_COMMAND} -E env
        LD_LIBRARY_PATH=${GDAL_INCLUDE_DIR}/../lib:$ENV{LD_LIBRARY_PATH}
        ${GDAL_TRANSLATE} -quiet -ot int32 -of GTiff
            ${CMAKE_CURRENT_SOURCE_DIR}/soil.asc
            ${CMAKE_CURRENT_BINARY_DIR}/unreadable.tif
    COMMAND chmod 222 ${CMAKE_CURRENT_BINARY_DIR}/unreadable.tif
)
list(APPEND outputs ${CMAKE_CURRENT_BINARY_DIR}/unreadable.tif)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/soil.tif
    COMMAND ${CMAKE_COMMAND} -E env
        LD_LIBRARY_PATH=${GDAL_INCLUDE_DIR}/../lib:$ENV{LD_LIBRARY_PATH}
        ${GDAL_TRANSLATE} -quiet -ot int32 -of GTiff
            ${CMAKE_CURRENT_SOURCE_DIR}/soil.asc
            ${CMAKE_CURRENT_BINARY_DIR}/soil.tif
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/soil.asc
)
list(APPEND outputs ${CMAKE_CURRENT_BINARY_DIR}/soil.tif)


add_custom_target(
    io_gdal_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    read_test
)
add_unit_tests(
    SCOPE io_gdal
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_io_core_objects>
        $<TARGET_OBJECTS:fern_io_gdal_objects>
    LINK_LIBRARIES
        fern_feature
        ${GDAL_LIBRARIES}
    DEPENDENCIES
        io_gdal_test_data
)
