add_test_conditionally(test)

set(SOURCES
    fern_client
    fern_dataset
    fern_driver
    hdf5_client
    hdf5_dataset
    hdf5_file
    hdf5_group
    hdf5_type_class_traits
    hdf5_type_traits
    utils
)
add_object_library(
    TARGET fern_language_io_fern_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
