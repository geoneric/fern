ADD_SUBDIRECTORY(test)

SET(SOURCES
    hdf5_client
    hdf5_dataset
    hdf5_dataset_driver
)
ADD_LIBRARY(geoneric_io_hdf5
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_io_hdf5
    geoneric_io_core
    geoneric_core
    ${HDF5_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
)
