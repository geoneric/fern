add_test_conditionally(test)

set(SOURCES
    gdal_client
    gdal_dataset
    gdal_driver
    gdal_data_type_traits;
)
add_object_library(
    TARGET fern_language_io_gdal_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
