# add_test_conditionally(test)

set(SOURCES
    # gdal_client
    # gdal_dataset
    # gdal_driver
    gdal_data_type_traits;
)
add_library_and_object_library(fern_io_gdal ${SOURCES})
target_link_libraries(fern_io_gdal
    # fern_io_core
    # fern_feature_visitor
    # fern_feature_core
    ${GDAL_LIBRARIES}
)
