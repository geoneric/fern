ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(core)
ADD_SUBDIRECTORY(gdal)
ADD_SUBDIRECTORY(geoneric)
# ADD_SUBDIRECTORY(hdf5)
# ADD_SUBDIRECTORY(netcdf)
# ADD_SUBDIRECTORY(ogr)
# ADD_SUBDIRECTORY(uncertml2)

SET(SOURCES
    drivers
    io_client
)
ADD_LIBRARY(geoneric_io
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_io
    geoneric_io_gdal
    geoneric_io_geoneric
)
