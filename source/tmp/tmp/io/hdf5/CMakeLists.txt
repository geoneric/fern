ADD_SUBDIRECTORY(test)

SET(SOURCES
    hdf5_client
    hdf5_dataset
    hdf5_dataset_driver
)
ADD_LIBRARY(fern_io_hdf5
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(fern_io_hdf5
    fern_io_core
    fern_core
    ${HDF5_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
)
