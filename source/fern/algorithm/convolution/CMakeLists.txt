add_subdirectory(neighborhood)
if(FERN_TEST)
    add_subdirectory(test)
endif()

set(SOURCES
    # index_ranges
)
# add_library_and_object_library(fern_algorithm_core ${SOURCES})
