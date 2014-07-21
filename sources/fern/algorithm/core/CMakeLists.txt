ADD_SUBDIRECTORY(test)

SET(SOURCES
    index_range
    index_ranges
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_algorithm_core ${SOURCES})
TARGET_LINK_LIBRARIES(fern_algorithm_core
    stdc++
)
IF(FERN_ALGORITHM_ONLY)
    INSTALL(
        TARGETS fern_algorithm_core
        EXPORT fern_targets
        ARCHIVE DESTINATION lib
    )
ENDIF()
