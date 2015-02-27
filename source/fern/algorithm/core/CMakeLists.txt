if(FERN_TEST)
    add_subdirectory(test)
endif()

set(SOURCES
    index_range
    index_ranges
)
add_library_and_object_library(fern_algorithm_core ${SOURCES})
target_link_libraries(fern_algorithm_core
    stdc++
)
if(FERN_ALGORITHM)
    install(
        TARGETS fern_algorithm_core
        EXPORT fern_targets
        ARCHIVE DESTINATION lib
    )
endif()
