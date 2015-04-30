add_test_conditionally(test)

set(SOURCES
    attribute
    domain
    feature
)

add_library_and_object_library(fern_language_feature_core ${SOURCES})
target_link_libraries(fern_language_feature_core
    fern_core
)
