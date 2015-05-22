set(SOURCES
    borrowed_reference
    client
    exception
    owned_reference
    string
)
add_object_library(
    TARGET fern_language_python_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
