add_test_conditionally(test)

set(SOURCES
    array
)
add_object_library(
    TARGET fern_feature_core_objects
    SOURCES ${SOURCES}
    LIBRARY fern_feature)
