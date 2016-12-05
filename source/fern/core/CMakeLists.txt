add_test_conditionally(test)

set(SOURCES
    data_name
    data_types
    detail/function_wrapper
    exception
    expression_type
    io_error
    join_threads
    messages
    parse_error
    path
    script_error
    stack
    stopwatch
    string
    thread_pool
    type_traits
    validate_error
    value_types
)
add_library(fern_core_objects OBJECT ${SOURCES})


if(FERN_CORE_DYN_LINK)
    add_library(fern_core SHARED
        $<TARGET_OBJECTS:fern_core_objects>
    )
else()
    add_library(fern_core STATIC
        $<TARGET_OBJECTS:fern_core_objects>
    )
endif()
set_target_properties(fern_core
    PROPERTIES
        COMPILE_FLAGS "-DFERN_CORE_SOURCE"
)
target_link_libraries(fern_core
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_TIMER_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
)
if(UNIX AND (
        ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR
        ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang"))
    target_link_libraries(fern_core
        pthread
    )
endif()
install(
    TARGETS fern_core
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
