add_test_conditionally(test)

set(SOURCES
    back_end
)
add_object_library(
    TARGET fern_language_back_end_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
