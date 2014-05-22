IF(FERN_ALGORITHM_ONLY)
    ADD_SUBDIRECTORY(core)

    # Install all header for now. Otherwise we have to keep tweaking the
    # install rules.
    INSTALL(
        DIRECTORY core
        DESTINATION include/fern/feature
        FILES_MATCHING PATTERN "*.h"
    )
ELSE()
    ADD_SUBDIRECTORY(core)
    ADD_SUBDIRECTORY(visitor)
ENDIF()
