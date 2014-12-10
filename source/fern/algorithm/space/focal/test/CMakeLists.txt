SET(TEST_NAMES
    slope_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE algorithm_spatial_focal
        NAME ${NAME}
        LINK_LIBRARIES fern_core fern_algorithm_core fern_feature_core)
ENDFOREACH()
