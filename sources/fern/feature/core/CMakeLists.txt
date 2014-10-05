ADD_SUBDIRECTORY(test)

IF(FERN_ALGORITHM)
    SET(SOURCES
        array
    )
ENDIF()

### ELSE()
###     SET(SOURCES
###         array
###         attribute
###         domain
###         feature
###     )
### ENDIF()

ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_feature_core ${SOURCES})
TARGET_LINK_LIBRARIES(fern_feature_core
    fern_core
)
IF(FERN_ALGORITHM)
    INSTALL(
        TARGETS fern_feature_core
        EXPORT fern_targets
        ARCHIVE DESTINATION lib
    )
ENDIF()
