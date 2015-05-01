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
add_library_and_object_library(fern_language_io ${SOURCES})
target_link_libraries(fern_language_io
    fern_language_io_gdal
    fern_language_io_fern
)
