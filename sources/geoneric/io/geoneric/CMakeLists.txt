ADD_SUBDIRECTORY(test)

SET(SOURCES
    geoneric_client
    geoneric_dataset
    geoneric_driver
    hdf5_client
    utils
)
ADD_LIBRARY(geoneric_io_geoneric
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_io_geoneric
    geoneric_io_core
    # geoneric_feature_visitor
    geoneric_feature_core
    ${HDF5_LIBRARIES}
)
