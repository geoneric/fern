set(TEST_NAMES
    gradient_test
    laplacian_test
    lax_test
)
add_unit_tests(
    SCOPE algorithm_algebra_vector
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_algorithm_policy_objects>
        $<TARGET_OBJECTS:fern_algorithm_core_objects>
    LINK_LIBRARIES
        fern_feature
)
