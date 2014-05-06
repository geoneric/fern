ADD_SUBDIRECTORY(test)

SET(SOURCES
    index_range
    index_ranges
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_algorithm_core ${SOURCES})
TARGET_LINK_LIBRARIES(fern_algorithm_core
    stdc++
)
