set(TEST_NAMES
  square_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_convolution_neighborhood
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy fern_algorithm_core
            fern_feature_core)
endforeach()
