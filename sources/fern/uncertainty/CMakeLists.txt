ADD_SUBDIRECTORY(test)

SET(SOURCES
    distribution
    normal_distribution
    uncertainty
)
ADD_LIBRARY(fern_uncertainty
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(fern_uncertainty
)
