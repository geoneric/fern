add_test_conditionally(test)

set(SOURCES
    dataset
    dataset_handle
    gdal_client
    gdal_value_type_traits
)
add_object_library(
    TARGET fern_io_gdal_objects
    SOURCES ${SOURCES}
    LIBRARY fern_io)
