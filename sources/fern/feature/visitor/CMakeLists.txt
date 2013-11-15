ADD_SUBDIRECTORY(test)

SET(SOURCES
    attribute_visitor
    attribute_type_visitor
)
ADD_LIBRARY(fern_feature_visitor
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(fern_feature_visitor
    fern_core
)
