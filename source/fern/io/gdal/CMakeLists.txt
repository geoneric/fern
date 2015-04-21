add_test_conditionally(test)

set(SOURCES
    dataset
    dataset_handle
    gdal_client
    gdal_value_type_traits
)
add_library_and_object_library(fern_io_gdal ${SOURCES})
target_link_libraries(fern_io_gdal
    ${GDAL_LIBRARIES}
)
