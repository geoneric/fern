add_test_conditionally(test)

set(SOURCES
    algebra_parser
)
add_object_library(
    TARGET fern_language_script_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
