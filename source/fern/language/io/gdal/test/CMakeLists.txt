copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/raster-1.asc
    DESTINATION_FILENAME raster-1.asc
    DESTINATION_FILE_PATHNAMES_LIST outputs)
copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/raster-2.asc
    DESTINATION_FILENAME raster-2.asc
    DESTINATION_FILE_PATHNAMES_LIST outputs)
add_custom_target(
    language_io_gdal_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    gdal_dataset_test
)
add_unit_tests(
    SCOPE language_io_gdal
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_language_feature_core_objects>
        $<TARGET_OBJECTS:fern_language_feature_visitor_objects>
        $<TARGET_OBJECTS:fern_language_io_objects>
        $<TARGET_OBJECTS:fern_language_io_core_objects>
        $<TARGET_OBJECTS:fern_language_io_fern_objects>
        $<TARGET_OBJECTS:fern_language_io_gdal_objects>
    LINK_LIBRARIES
        fern_feature
        ${GDAL_LIBRARIES}
        ${HDF5_LIBRARIES}
    DEPENDENCIES
        language_io_gdal_test_data
)
