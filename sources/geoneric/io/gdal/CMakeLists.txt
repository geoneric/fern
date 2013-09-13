ADD_SUBDIRECTORY(test)

SET(SOURCES
    attribute
    domain
    feature
)
ADD_LIBRARY(geoneric_io_gdal
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_io_gdal
    geoneric_io_core
    # ${GDAL_LIBRARIES}
)
