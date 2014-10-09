ADD_SUBDIRECTORY(test)

SET(SOURCES
    attribute
    domain
    feature
    # feature_domain_value
    scalar_domain
    value
)
ADD_LIBRARY(fern_feature
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(fern_feature
    fern_operation_core
    fern_core
)
