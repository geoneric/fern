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
