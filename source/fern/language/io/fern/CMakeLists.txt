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
add_library_and_object_library(fern_language_io_fern ${SOURCES})
target_link_libraries(fern_language_io_fern
    fern_language_io_core
    fern_language_feature_visitor
    fern_language_feature_core
    ${HDF5_LIBRARIES}
)
