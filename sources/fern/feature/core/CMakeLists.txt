ADD_SUBDIRECTORY(test)

SET(SOURCES
    array
    attribute
    domain
    feature
)
ADD_LIBRARY(geoneric_feature_core
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_feature_core
    geoneric_core
)
