add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/constant-1.h5
    COMMAND $<TARGET_FILE:fern> execute
        ${FERN_DATA_DIR}/test/constant-1.frn
    DEPENDS fern ${FERN_DATA_DIR}/test/constant-1.frn
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/constant-1.h5
)


add_custom_target(
    language_io_fern_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    fern_dataset_test
)
add_unit_tests(
    SCOPE language_io_fern
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
        language_io_fern_test_data
)
