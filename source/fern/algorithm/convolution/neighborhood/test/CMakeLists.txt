SET(TEST_NAMES
  square_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE algorithm_convolution_neighborhood
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy fern_algorithm_core
            fern_feature_core)
ENDFOREACH()
