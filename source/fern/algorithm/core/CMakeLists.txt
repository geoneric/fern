add_test_conditionally(test)

set(SOURCES
    index_range
    index_ranges
)
add_object_library(
    TARGET fern_algorithm_core_objects
    SOURCES ${SOURCES}
    LIBRARY fern_algorithm)
