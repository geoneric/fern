add_test_conditionally(test)

add_executable(tutorial_divide-1
    tutorial_divide-1
)
# target_include_directories(tutorial_divide-1
#     ${CMAKE_SOURCE_DIR}/source
#     ${CMAKE_BINARY_DIR}/source
# )
target_link_libraries(tutorial_divide-1
    PRIVATE
        fern::algorithm
)

add_executable(tutorial_divide-2
    tutorial_divide-2
)
target_link_libraries(tutorial_divide-2
    PRIVATE
        fern::algorithm
)

add_executable(tutorial_divide-3
    tutorial_divide-3
)
target_link_libraries(tutorial_divide-3
    PRIVATE
        fern::algorithm
        fern::feature
)

add_executable(tutorial_divide-4
    tutorial_divide-4
)
target_link_libraries(tutorial_divide-4
    PRIVATE
        fern::algorithm
        fern::feature
)

add_executable(tutorial_divide-5
    tutorial_divide-5
)
target_link_libraries(tutorial_divide-5
    PRIVATE
        fern::algorithm
        fern::feature
)
