add_test_conditionally(test)

set(SOURCES
    execution_policy
    parallel_execution_policy
)
add_object_library(
    TARGET fern_algorithm_policy_objects
    SOURCES ${SOURCES}
    LIBRARY fern_algorithm)
