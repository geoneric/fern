add_test_conditionally(test)

add_executable(add_two_rasters
    add_two_rasters.cc
)
target_link_libraries(add_two_rasters
    fern_algorithm
    fern_feature
)
