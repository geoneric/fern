add_test_conditionally(test)

set(SOURCES
    attribute
    domain
    feature
)
add_object_library(
    TARGET fern_language_feature_core_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
