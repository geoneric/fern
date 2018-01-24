add_subdirectory(neighborhood)
add_test_conditionally(test)

if(FERN_BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
