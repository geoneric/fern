ADD_SUBDIRECTORY(test)

SET(SOURCES
    ogr_client
    ogr_dataset
    ogr_dataset_driver
    ogr_feature_layer
)
ADD_LIBRARY(fern_io_ogr
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(fern_io_ogr
    fern_io_core
    ${GDAL_LIBRARIES}
)
