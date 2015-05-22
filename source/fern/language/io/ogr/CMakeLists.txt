add_test_conditionally(test)

set(SOURCES
    dataset_driver
    ogr_client
    ogr_dataset
    ogr_dataset_driver
    ogr_feature_layer
)
add_object_library(
    TARGET fern_language_io_ogr_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
# target_link_libraries(fern_language_io_ogr
#     fern_language_io_core
#     ${GDAL_LIBRARIES}
# )
