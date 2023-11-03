add_test_conditionally(test)

add_library(fern_core
    data_name.cc
    data_types.cc
    exception.cc
    expression_type.cc
    io_error.cc
    join_threads.cc
    messages.cc
    parse_error.cc
    path.cc
    script_error.cc
    stack.cc
    # stopwatch.cc
    string.cc
    thread_pool.cc
    type_traits.cc
    validate_error.cc
    value_types.cc
)

add_library(fern::core ALIAS fern_core)

target_include_directories(fern_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(fern_core
    # PUBLIC
    #     Boost::timer
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
