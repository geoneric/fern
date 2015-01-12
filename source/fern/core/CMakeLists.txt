ADD_SUBDIRECTORY(test)

SET(SOURCES
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
IF(FERN_COMPILER_HAS_REGEX)
    # Skip modules that depend on String::split, which is not included if
    # the compiler doesn't support std::regex.
    SET(SOURCES
        ${SOURCES}
        data_name
        path
    )
ENDIF()
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_core ${SOURCES})
TARGET_LINK_LIBRARIES(fern_core
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_TIMER_LIBRARY}
    stdc++
)
IF(FERN_ALGORITHM)
    INSTALL(
        TARGETS fern_core
        EXPORT fern_targets
        ARCHIVE DESTINATION lib
    )
ENDIF()
