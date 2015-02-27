if(FERN_TEST)
    add_subdirectory(test)
endif()

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
