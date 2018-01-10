add_test_conditionally(test)

add_library(fern_core
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
    stopwatch
    string
    thread_pool
    type_traits
    validate_error
    value_types
)

add_library(fern::core ALIAS fern_core)

target_include_directories(fern_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

target_link_libraries(fern_core
    PUBLIC
        Boost::filesystem
        Boost::timer
    PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:Threads::Threads>
)

install(
    TARGETS fern_core
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
