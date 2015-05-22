add_test_conditionally(test)

set(SOURCES
    dataset
)
add_object_library(
    TARGET fern_io_netcdf_coards_objects
    SOURCES ${SOURCES}
    LIBRARY fern_io)
