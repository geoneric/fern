add_subdirectory(test)

if(FERN_ALGORITHM)
    set(SOURCES
        array
    )
endif()

### else()
###     set(SOURCES
###         array
###         attribute
###         domain
###         feature
###     )
### endif()

add_library_and_object_library(fern_feature_core ${SOURCES})
target_link_libraries(fern_feature_core
    fern_core
)
if(FERN_ALGORITHM)
    install(
        TARGETS fern_feature_core
        EXPORT fern_targets
        ARCHIVE DESTINATION lib
    )
endif()
