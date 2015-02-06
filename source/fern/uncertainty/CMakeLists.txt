add_subdirectory(test)

set(SOURCES
    distribution
    normal_distribution
    uncertainty
)
add_library(fern_uncertainty
    ${SOURCES}
)
target_link_libraries(fern_uncertainty
)
