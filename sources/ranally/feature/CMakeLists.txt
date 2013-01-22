ADD_SUBDIRECTORY(test)

SET(SOURCES
    attribute
    domain
    feature
    # feature_domain_value
    scalar_domain
    value
)
ADD_LIBRARY(ranally_feature
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_feature
    ranally_core
)
