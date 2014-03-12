ADD_SUBDIRECTORY(test)

SET(SOURCES
    data_name
    data_types
    exception
    expression_type
    io_error
    join_threads
    messages
    parse_error
    path
    script_error
    stack
    string
    thread_pool
    type_traits
    validate_error
    value_types
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_core ${SOURCES})
TARGET_LINK_LIBRARIES(fern_core
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${ICU_LIBRARIES}
)
