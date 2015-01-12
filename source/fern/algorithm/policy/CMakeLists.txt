ADD_SUBDIRECTORY(test)

SET(SOURCES
    execution_policy
    parallel_execution_policy
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_algorithm_policy ${SOURCES})
TARGET_LINK_LIBRARIES(fern_algorithm_policy
    fern_core
    stdc++
)
