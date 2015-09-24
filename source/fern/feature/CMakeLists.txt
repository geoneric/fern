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

if(FERN_FEATURE_DYN_LINK)
    add_library(fern_feature SHARED
        ${FERN_FEATURE_OBJECT_LIBRARIES}
    )
else()
    add_library(fern_feature STATIC
        ${FERN_FEATURE_OBJECT_LIBRARIES}
    )
endif()
set_target_properties(fern_algorithm
    PROPERTIES
        COMPILE_FLAGS "-DFERN_FEATURE_SOURCE"
)
target_link_libraries(fern_feature
    fern_core
)
install(
    TARGETS fern_feature
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
