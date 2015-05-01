copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/valid-1.ran
    DESTINATION_FILENAME valid-1.ran
    DESTINATION_FILE_PATHNAMES_LIST outputs)
copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/valid-1.ran
    DESTINATION_FILENAME valid-1_unreadable.ran
    PERMISSIONS WRITE_ONLY
    DESTINATION_FILE_PATHNAMES_LIST outputs)
copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/valid-2.ran
    DESTINATION_FILENAME valid-2.ran
    DESTINATION_FILE_PATHNAMES_LIST outputs)
copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/invalid-1.ran
    DESTINATION_FILENAME invalid-1.ran
    DESTINATION_FILE_PATHNAMES_LIST outputs)
copy_test_file(
    SOURCE_FILE_PATHNAME ${FERN_DATA_DIR}/test/raster-1.asc
    DESTINATION_FILENAME raster-1.asc
    DESTINATION_FILE_PATHNAMES_LIST outputs)


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
    language_interpreter_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    execute_visitor_test
    interpreter_test
)
add_unit_tests(
    SCOPE language_interpreter
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_interpreter
    DEPENDENCIES language_interpreter_test_data)
