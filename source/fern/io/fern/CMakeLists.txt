add_test_conditionally(test)

set(SOURCES
    fern_client
    fern_dataset
    fern_driver
    hdf5_client
    hdf5_type_class_traits
    hdf5_type_traits
    utils
)
add_library_and_object_library(fern_io_fern ${SOURCES})
target_link_libraries(fern_io_fern
    fern_io_core
    fern_feature_visitor
    fern_feature_core
    ${HDF5_LIBRARIES}
)
