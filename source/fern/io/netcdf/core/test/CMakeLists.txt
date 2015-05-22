add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/earth.nc
    COMMAND ${NCGEN} -k 1 -o ${CMAKE_CURRENT_BINARY_DIR}/earth.nc
        ${CMAKE_CURRENT_SOURCE_DIR}/earth.cdl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/earth.cdl
)
list(APPEND outputs ${CMAKE_CURRENT_BINARY_DIR}/earth.nc)


add_custom_target(
    io_netcdf_core_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    attribute_test
    variable_test
)
add_unit_tests(
    SCOPE io_netcdf_core
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_io_core_objects>
        $<TARGET_OBJECTS:fern_io_netcdf_core_objects>
    LINK_LIBRARIES
        fern_core
        ${NETCDF_LIBRARIES}
    DEPENDENCIES
        io_netcdf_core_test_data
)
