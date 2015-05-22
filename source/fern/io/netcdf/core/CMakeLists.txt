add_test_conditionally(test)

set(SOURCES
    attribute
    dataset
    dataset_handle
    variable
)
add_object_library(
    TARGET fern_io_netcdf_core_objects
    SOURCES ${SOURCES}
    LIBRARY fern_io)
