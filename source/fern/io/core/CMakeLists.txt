add_test_conditionally(test)

set(SOURCES
    file.cc
)
add_object_library(
    TARGET fern_io_core_objects
    SOURCES ${SOURCES}
    LIBRARY fern_io)
