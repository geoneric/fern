add_test_conditionally(test)

set(SOURCES
    dataset_driver
    ogr_client
    ogr_dataset
    ogr_dataset_driver
    ogr_feature_layer
)
add_library(fern_language_io_ogr
    ${SOURCES}
)
target_link_libraries(fern_language_io_ogr
    fern_language_io_core
    ${GDAL_LIBRARIES}
)
