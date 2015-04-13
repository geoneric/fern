if(FERN_BUILD_ALGORITHM)
    add_subdirectory(core)

    # Install all header for now. Otherwise we have to keep tweaking the
    # install rules.
    install(
        DIRECTORY core
        DESTINATION include/fern/feature
        CONFIGURATIONS Release
        FILES_MATCHING PATTERN "*.h"
    )
endif()

### ADD_SUBDIRECTORY(visitor)
