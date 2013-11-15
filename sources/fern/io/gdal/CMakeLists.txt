ADD_SUBDIRECTORY(test)

SET(SOURCES
    gdal_client
    gdal_dataset
    gdal_driver
    gdal_data_type_traits;
)
ADD_LIBRARY(fern_io_gdal
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(fern_io_gdal
    fern_io_core
    fern_feature_visitor
    fern_feature_core
    ${GDAL_LIBRARIES}
)
