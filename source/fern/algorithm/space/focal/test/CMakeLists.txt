set(TEST_NAMES
    slope_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_spatial_focal
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy fern_core fern_algorithm_core
            fern_feature_core)
endforeach()
