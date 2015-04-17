add_test_conditionally(test)

set(SOURCES
    attribute
    dataset
    dataset_handle
    variable
)
add_library_and_object_library(fern_io_netcdf_core ${SOURCES})
target_link_libraries(fern_io_netcdf_core
    stdc++
)
