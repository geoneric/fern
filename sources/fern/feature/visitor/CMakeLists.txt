ADD_SUBDIRECTORY(test)

SET(SOURCES
    attribute_visitor
    attribute_type_visitor
)
ADD_LIBRARY(geoneric_feature_visitor
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_feature_visitor
    geoneric_core
)
