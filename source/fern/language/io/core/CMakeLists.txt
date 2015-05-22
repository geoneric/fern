add_test_conditionally(test)

set(SOURCES
    dataset
    driver
    file
)
add_object_library(
    TARGET fern_language_io_core_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
