add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/unreadable.nc
    COMMAND ${NCGEN} -k 1 -o ${CMAKE_CURRENT_BINARY_DIR}/unreadable.nc
        ${CMAKE_CURRENT_SOURCE_DIR}/earth.cdl
    COMMAND chmod 222 ${CMAKE_CURRENT_BINARY_DIR}/unreadable.nc
)
list(APPEND outputs ${CMAKE_CURRENT_BINARY_DIR}/unreadable.nc)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/invalid_netcdf.nc
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/invalid_netcdf.nc
        ${CMAKE_CURRENT_BINARY_DIR}/invalid_netcdf.nc
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/invalid_netcdf.nc
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/invalid_netcdf_coards.nc
    COMMAND ${NCGEN} -k 1 -o
        ${CMAKE_CURRENT_BINARY_DIR}/invalid_netcdf_coards.nc
        ${CMAKE_CURRENT_SOURCE_DIR}/invalid_netcdf_coards.cdl
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/invalid_netcdf_coards.nc
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/earth.nc
    COMMAND ${NCGEN} -k 1 -o ${CMAKE_CURRENT_BINARY_DIR}/earth.nc
        ${CMAKE_CURRENT_SOURCE_DIR}/earth.cdl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/earth.cdl
)
list(APPEND outputs ${CMAKE_CURRENT_BINARY_DIR}/earth.nc)


add_custom_target(
    io_netcdf_coards_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    read_test
)
add_unit_tests(
    SCOPE io_netcdf_coards
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_io_core_objects>
        $<TARGET_OBJECTS:fern_io_netcdf_coards_objects>
        $<TARGET_OBJECTS:fern_io_netcdf_core_objects>
    LINK_LIBRARIES
        fern_core
        ${NETCDF_LIBRARIES}
    DEPENDENCIES
        io_netcdf_coards_test_data
)
