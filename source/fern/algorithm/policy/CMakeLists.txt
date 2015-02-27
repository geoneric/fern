if(FERN_TEST)
    add_subdirectory(test)
endif()

set(SOURCES
    execution_policy
    parallel_execution_policy
)
add_library_and_object_library(fern_algorithm_policy ${SOURCES})
target_link_libraries(fern_algorithm_policy
    fern_core
    stdc++
)
if(FERN_ALGORITHM)
    install(
        TARGETS fern_algorithm_policy
        EXPORT fern_targets
        ARCHIVE DESTINATION lib
    )
endif()
