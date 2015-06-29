set(TEST_NAMES
    and_test
    defined_test
    not_test
    or_test
    xor_test
)
add_unit_tests(
    SCOPE algorithm_algebra_boole
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_algorithm_policy_objects>
    LINK_LIBRARIES
        fern_core
)
