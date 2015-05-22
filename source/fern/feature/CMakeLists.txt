add_subdirectory(core)

if(FERN_BUILD_ALGORITHM)
    # Install all header for now. Otherwise we have to keep tweaking the
    # install rules.
    install(
        DIRECTORY core
        DESTINATION include/fern/feature
        CONFIGURATIONS Release
        FILES_MATCHING PATTERN "*.h"
    )
endif()


get_property(FERN_FEATURE_OBJECT_LIBRARIES GLOBAL
    PROPERTY FERN_FEATURE_OBJECT_LIBRARIES)
add_library(fern_feature SHARED
    ${FERN_FEATURE_OBJECT_LIBRARIES}
)
target_link_libraries(fern_feature
    fern_core
    # ${FERN_EXTERNAL_LIBRARIES}
)
install(
    TARGETS fern_feature
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
