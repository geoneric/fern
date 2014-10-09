ADD_SUBDIRECTORY(test)

SET(SOURCES
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
    thread_client
    thread_pool
    type_traits
    validate_error
    value_types
)
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
