set(TEST_NAMES
    acos_test
    asin_test
    atan_test
    cos_test
    sin_test
    tan_test
)
add_unit_tests(
    SCOPE algorithm_trigonometry
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_algorithm_policy_objects>
    LINK_LIBRARIES
        fern_core
)
