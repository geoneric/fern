add_test_conditionally(test)

set(SOURCES
    ogr_client
    ogr_dataset
    ogr_dataset_driver
    ogr_feature_layer
)
add_library(fern_io_ogr
    ${SOURCES}
)
target_link_libraries(fern_io_ogr
    fern_io_core
    ${GDAL_LIBRARIES}
)
