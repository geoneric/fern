# ADD_SUBDIRECTORY(core)
# ADD_SUBDIRECTORY(hdf5)
# ADD_SUBDIRECTORY(netcdf)
# ADD_SUBDIRECTORY(ogr)
# ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(uncertml2)

# SET(SOURCES
#     attribute
#     dataset
#     dataset_driver
#     domain
#     feature
#     # hdf5_client
#     # hdf5_dataset
#     # hdf5_dataset_driver
#     # ogr_client
#     # ogr_dataset
#     # ogr_dataset_driver
#     # ogr_feature_layer
#     point_attribute
#     point_domain
#     point_feature
#     point_value
#     polygon_attribute
#     polygon_domain
#     polygon_feature
#     polygon_value
#     raster_attribute
#     raster_value
#     spatial_domain
#     spatio_temporal_domain
#     temporal_domain
#     util
#     value
# )
# ADD_LIBRARY(ranally_io
#     ${SOURCES}
# )
# TARGET_LINK_LIBRARIES(ranally_io
#     ranally_core
#     # ${HDF5_LIBRARIES}
#     # ${GDAL_LIBRARIES}
#     ${Boost_FILESYSTEM_LIBRARY}
#     ${Boost_SYSTEM_LIBRARY}
# )
