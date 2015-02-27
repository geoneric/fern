include_directories(
    ${CMAKE_SOURCE_DIR}/source
    ${CMAKE_BINARY_DIR}/source
)
if(FERN_TEST)
    add_subdirectory(test)
endif()

add_executable(tutorial_divide-1
    tutorial_divide-1
)
target_link_libraries(tutorial_divide-1
    fern_algorithm_policy
    stdc++
)

add_executable(tutorial_divide-2
    tutorial_divide-2
)
target_link_libraries(tutorial_divide-2
    fern_algorithm_policy
    fern_algorithm_core
)

add_executable(tutorial_divide-3
    tutorial_divide-3
)
target_link_libraries(tutorial_divide-3
    fern_algorithm_policy
    fern_algorithm_core
    fern_feature_core
)

add_executable(tutorial_divide-4
    tutorial_divide-4
)
target_link_libraries(tutorial_divide-4
    fern_algorithm_policy
    fern_algorithm_core
    fern_feature_core
)

add_executable(tutorial_divide-5
    tutorial_divide-5
)
target_link_libraries(tutorial_divide-5
    fern_algorithm_policy
    fern_algorithm_core
    fern_feature_core
)
