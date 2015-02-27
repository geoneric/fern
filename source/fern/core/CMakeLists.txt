if(FERN_TEST)
    add_subdirectory(test)
endif()

set(SOURCES
    data_types
    detail/function_wrapper
    exception
    expression_type
    io_error
    join_threads
    messages
    parse_error
    script_error
    stack
    stopwatch
    string
    thread_pool
    type_traits
    validate_error
    value_types
)
if(FERN_COMPILER_HAS_REGEX)
    # Skip modules that depend on String::split, which is not included if
    # the compiler doesn't support std::regex.
    set(SOURCES
        ${SOURCES}
        data_name
        path
    )
endif()
add_library_and_object_library(fern_core ${SOURCES})
target_link_libraries(fern_core
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_TIMER_LIBRARY}
    stdc++
)
if(FERN_ALGORITHM)
    install(
        TARGETS fern_core
        EXPORT fern_targets
        ARCHIVE DESTINATION lib
    )
endif()
