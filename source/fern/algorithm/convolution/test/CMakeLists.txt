SET(TEST_NAMES
  convolve_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(algorithm_convolution ${NAME}
        fern_algorithm_core fern_feature_core)
ENDFOREACH()
