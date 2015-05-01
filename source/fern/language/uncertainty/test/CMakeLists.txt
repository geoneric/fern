set(TEST_NAMES
    normal_distribution_test
)
add_unit_tests(
    SCOPE language_uncertainty
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_uncertainty)
