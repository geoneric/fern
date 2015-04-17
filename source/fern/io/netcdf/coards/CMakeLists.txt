add_test_conditionally(test)

set(SOURCES
    dataset
)
add_library_and_object_library(fern_io_netcdf_coards ${SOURCES})
target_link_libraries(fern_io_netcdf_coards
    stdc++
)
