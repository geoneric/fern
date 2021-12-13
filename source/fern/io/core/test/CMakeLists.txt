add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/unreadable.txt
    COMMAND ${CMAKE_COMMAND} -E touch
        ${CMAKE_CURRENT_BINARY_DIR}/unreadable.txt
    COMMAND chmod 222 ${CMAKE_CURRENT_BINARY_DIR}/unreadable.txt
)
list(APPEND outputs ${CMAKE_CURRENT_BINARY_DIR}/unreadable.txt)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/unwritable.txt
    COMMAND ${CMAKE_COMMAND} -E touch
        ${CMAKE_CURRENT_BINARY_DIR}/unwritable.txt
    COMMAND chmod 444 ${CMAKE_CURRENT_BINARY_DIR}/unwritable.txt
)
list(APPEND outputs ${CMAKE_CURRENT_BINARY_DIR}/unwritable.txt)


add_custom_target(
    io_core_test_data
    DEPENDS ${outputs}
)


set(TEST_NAMES
    file_test.cc
)
add_unit_tests(
    SCOPE io_core
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_io_core_objects>
    LINK_LIBRARIES
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
    DEPENDENCIES
        io_core_test_data
)
