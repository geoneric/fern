add_subdirectory(accumulator)
add_subdirectory(algebra)
add_subdirectory(convolution)
add_subdirectory(core)
add_subdirectory(policy)
add_subdirectory(space)
add_subdirectory(statistic)
add_test_conditionally(test)
add_subdirectory(trigonometry)

add_library(fern_algorithm
    core/index_range
    core/index_ranges
    policy/execution_policy
    policy/parallel_execution_policy
)

add_library(fern::algorithm ALIAS fern_algorithm)

target_include_directories(fern_algorithm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

target_link_libraries(fern_algorithm
    PUBLIC
        fern::core
)

install(
    TARGETS fern_algorithm
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
