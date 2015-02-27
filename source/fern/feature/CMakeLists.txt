if(FERN_ALGORITHM)
    add_subdirectory(core)

    # Install all header for now. Otherwise we have to keep tweaking the
    # install rules.
    install(
        DIRECTORY core
        DESTINATION include/fern/feature
        FILES_MATCHING PATTERN "*.h"
        CONFIGURATIONS Release
    )
endif()

### ADD_SUBDIRECTORY(visitor)
