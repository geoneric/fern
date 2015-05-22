set(TEST_NAMES
  square_test
)
add_unit_tests(
    SCOPE algorithm_convolution_neighborhood
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_algorithm_policy_objects>
        $<TARGET_OBJECTS:fern_algorithm_core_objects>
    LINK_LIBRARIES
        fern_core
)
