ADD_SUBDIRECTORY(test)

SET(SOURCES
    array
    attribute
    dataset
    domain
    feature
    gdal_dataset
)
ADD_LIBRARY(geoneric_io_gdal
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_io_gdal
    geoneric_io_core
    ${GDAL_LIBRARIES}
)
