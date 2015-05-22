add_test_conditionally(test)

set(SOURCES
    distribution
    normal_distribution
    uncertainty
)
add_object_library(
    TARGET fern_language_uncertainty_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
