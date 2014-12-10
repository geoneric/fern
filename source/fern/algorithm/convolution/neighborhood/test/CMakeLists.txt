SET(TEST_NAMES
  square_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE algorithm_convolution_neighborhood
        NAME ${NAME}
        LINK_LIBRARIES fern_feature_core fern_algorithm_core)
ENDFOREACH()
