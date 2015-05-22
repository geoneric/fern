set(TEST_NAMES
    constant_test
    evaluate_visitor_test
    expression_test
    raster_test
    stream_visitor_test
)
add_unit_tests(
    SCOPE expression_tree
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_feature
)
