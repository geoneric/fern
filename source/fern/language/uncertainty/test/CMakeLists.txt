set(TEST_NAMES
    normal_distribution_test
)
add_unit_tests(
    SCOPE language_uncertainty
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_language_uncertainty_objects>
)
