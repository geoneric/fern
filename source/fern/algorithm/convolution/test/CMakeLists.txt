set(TEST_NAMES
  convolve_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_convolution
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy fern_algorithm_core
            fern_feature_core)
endforeach()
