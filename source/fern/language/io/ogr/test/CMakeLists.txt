set(TEST_NAMES
    ogr_dataset_driver_test
)
add_unit_tests(
    SCOPE language_io_ogr
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_io_ogr)
