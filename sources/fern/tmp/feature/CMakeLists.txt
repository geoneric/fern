ADD_SUBDIRECTORY(test)

SET(SOURCES
    attribute
    domain
    feature
    # feature_domain_value
    scalar_domain
    value
)
ADD_LIBRARY(geoneric_feature
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_feature
    geoneric_operation_core
    geoneric_core
)
