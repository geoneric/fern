add_test_conditionally(test)

set(SOURCES
    file
)
add_library_and_object_library(fern_io_core ${SOURCES})
target_link_libraries(fern_io_core
    stdc++
)
