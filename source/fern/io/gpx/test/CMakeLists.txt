copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/example-1.gpx
    DESTINATION_FILENAME example-1.gpx
    DESTINATION_FILE_PATHNAMES_LIST outputs)
copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/example-1.gpx
    DESTINATION_FILENAME unreadable.gpx
    PERMISSIONS WRITE_ONLY
    DESTINATION_FILE_PATHNAMES_LIST outputs)
copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/invalid.gpx
    DESTINATION_FILENAME invalid.gpx
    DESTINATION_FILE_PATHNAMES_LIST outputs)
add_custom_target(
    io_gpx_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    parse_1_0_test
)
add_unit_tests(
    SCOPE io_gpx
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_io_core_objects>
        $<TARGET_OBJECTS:fern_io_gpx_objects>
    LINK_LIBRARIES
        fern_core
        ${XercesC_LIBRARIES}
    DEPENDENCIES
        io_gpx_test_data
)
