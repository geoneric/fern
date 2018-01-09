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

add_library(fern_feature SHARED
    core/array
)

add_library(fern::feature ALIAS fern_feature)

target_include_directories(fern_feature
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

target_link_libraries(fern_feature
    PUBLIC
        fern::core
)

install(
    TARGETS fern_feature
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
