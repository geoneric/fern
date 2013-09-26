ADD_SUBDIRECTORY(test)

SET(SOURCES
    dataset
    gdal_dataset
)
ADD_LIBRARY(geoneric_io_gdal
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_io_gdal
    geoneric_io_core
    geoneric_feature_core
    ${GDAL_LIBRARIES}
)
