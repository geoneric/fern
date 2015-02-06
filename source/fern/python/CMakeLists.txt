set(SOURCES
    borrowed_reference
    client
    exception
    owned_reference
    string
)
add_library_and_object_library(fern_python ${SOURCES})
