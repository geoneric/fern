SET(TEST_NAMES
  square_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(algorithm_convolution_neighborhood ${NAME}
        fern_feature_core fern_algorithm_core)
ENDFOREACH()
