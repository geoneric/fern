INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}/source
    ${CMAKE_BINARY_DIR}/source
)
ADD_SUBDIRECTORY(test)

ADD_EXECUTABLE(tutorial_divide-1
    tutorial_divide-1
)
TARGET_LINK_LIBRARIES(tutorial_divide-1
    stdc++
)

ADD_EXECUTABLE(tutorial_divide-2
    tutorial_divide-2
)
TARGET_LINK_LIBRARIES(tutorial_divide-2
    fern_algorithm_core
)

ADD_EXECUTABLE(tutorial_divide-3
    tutorial_divide-3
)
TARGET_LINK_LIBRARIES(tutorial_divide-3
    fern_algorithm_core
    fern_feature_core
)

ADD_EXECUTABLE(tutorial_divide-4
    tutorial_divide-4
)
TARGET_LINK_LIBRARIES(tutorial_divide-4
    fern_algorithm_core
    fern_feature_core
)

ADD_EXECUTABLE(tutorial_divide-5
    tutorial_divide-5
)
TARGET_LINK_LIBRARIES(tutorial_divide-5
    fern_algorithm_core
    fern_feature_core
)
