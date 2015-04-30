add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/valid-1.ran
    COMMAND ${CMAKE_COMMAND} -E copy
        ${FERN_DATA_DIR}/test/valid-1.ran
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${FERN_DATA_DIR}/test/valid-1.ran
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/valid-1.ran
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/valid-1_unreadable.ran
    COMMAND ${CMAKE_COMMAND} -E copy
        ${FERN_DATA_DIR}/test/valid-1.ran
        ${CMAKE_CURRENT_BINARY_DIR}/valid-1_unreadable.ran
    COMMAND chmod a-r ${CMAKE_CURRENT_BINARY_DIR}/valid-1_unreadable.ran
    DEPENDS ${FERN_DATA_DIR}/test/valid-1.ran
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/valid-1_unreadable.ran
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/valid-2.ran
    COMMAND ${CMAKE_COMMAND} -E copy
        ${FERN_DATA_DIR}/test/valid-2.ran
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${FERN_DATA_DIR}/test/valid-2.ran
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/valid-2.ran
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/invalid-1.ran
    COMMAND ${CMAKE_COMMAND} -E copy
        ${FERN_DATA_DIR}/test/invalid-1.ran
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${FERN_DATA_DIR}/test/invalid-1.ran
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/invalid-1.ran
)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/raster-1.asc
    COMMAND ${CMAKE_COMMAND} -E copy
        ${FERN_DATA_DIR}/test/raster-1.asc
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${FERN_DATA_DIR}/test/raster-1.asc
)
list(APPEND outputs
    ${CMAKE_CURRENT_BINARY_DIR}/raster-1.asc
)


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
