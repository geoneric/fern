set(TEST_NAMES
    common_attributes_test
)
add_unit_tests(
    SCOPE wkt
    NAMES ${TEST_NAMES}
    # LINK_LIBRARIES
    #     fern_core
)
