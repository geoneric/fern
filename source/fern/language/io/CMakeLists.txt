add_test_conditionally(test)
add_subdirectory(core)
add_subdirectory(fern)
add_subdirectory(gdal)
# add_subdirectory(hdf5)
add_subdirectory(netcdf)
# add_subdirectory(ogr)
add_subdirectory(uncertml2)

set(SOURCES
    drivers
    io_client
)
add_object_library(
    TARGET fern_language_io_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
