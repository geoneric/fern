add_test_conditionally(test)

add_executable(add_two_rasters
    add_two_rasters
    raster
)
target_link_libraries(add_two_rasters
    fern_feature_core
    fern_algorithm_policy
    fern_algorithm_core
    fern_core
)
