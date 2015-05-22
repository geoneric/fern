add_subdirectory(core)
if(FERN_IO_WITH_GDAL)
    add_subdirectory(gdal)
endif()
if(FERN_IO_WITH_GPX)
    add_subdirectory(gpx)
    set(FERN_IO_WITH_XML true)
endif()
if(FERN_IO_WITH_NETCDF)
    add_subdirectory(netcdf)
endif()
if(FERN_IO_WITH_XML)
    add_subdirectory(xml)
endif()


get_property(FERN_IO_OBJECT_LIBRARIES GLOBAL PROPERTY FERN_IO_OBJECT_LIBRARIES)
add_library(fern_io SHARED
    ${FERN_IO_OBJECT_LIBRARIES}
)
target_link_libraries(fern_io
    fern_core
    ${GDAL_LIBRARIES}
    ${NETCDF_LIBRARIES}
    ${XercesC_LIBRARIES}
)
install(
    TARGETS fern_io
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
