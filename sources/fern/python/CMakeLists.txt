SET(SOURCES
    borrowed_reference
    client
    exception
    owned_reference
    string
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_python ${SOURCES})
