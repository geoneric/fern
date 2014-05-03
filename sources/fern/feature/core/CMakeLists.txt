ADD_SUBDIRECTORY(test)

SET(SOURCES
    array
    attribute
    domain
    feature
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_feature_core ${SOURCES})
TARGET_LINK_LIBRARIES(fern_feature_core
    fern_core
)
IF(FERN_ALGORITHM_ONLY)
    INSTALL(TARGETS fern_feature_core
        ARCHIVE DESTINATION lib
    )
ENDIF()
