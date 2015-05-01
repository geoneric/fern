add_test_conditionally(test)

set(SOURCES
    dataset
    driver
    file
)
add_library_and_object_library(fern_language_io_core ${SOURCES})
target_link_libraries(fern_language_io_core
    fern_core
)
