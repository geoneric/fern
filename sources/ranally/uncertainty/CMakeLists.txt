ADD_SUBDIRECTORY(test)

SET(SOURCES
    distribution
    normal_distribution
    uncertainty
)
ADD_LIBRARY(ranally_uncertainty
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_uncertainty
)
