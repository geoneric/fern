if(FERN_TEST)
    add_subdirectory(test)
endif()

set(SOURCES
    dataset
    driver
    file
)
add_library_and_object_library(fern_io_core ${SOURCES})
target_link_libraries(fern_io_core
    fern_core
)
