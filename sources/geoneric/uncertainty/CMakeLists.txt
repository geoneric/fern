ADD_SUBDIRECTORY(test)

SET(SOURCES
    distribution
    normal_distribution
    uncertainty
)
ADD_LIBRARY(geoneric_uncertainty
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_uncertainty
)
