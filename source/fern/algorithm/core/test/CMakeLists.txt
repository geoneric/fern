SET(TEST_NAMES
    binary_local_operation_test
    cast_test
    compress_test
    copy_test
    cover_test
    decompress_test
    if_test
    index_range_test
    index_ranges_test
    intersect_no_data_test
    fill_test
    nullary_local_operation_test
    offset_test
    result_type_test
    result_value_test
    unary_disaggregate_operation_test
    unary_local_operation_test
    unite_no_data_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE algorithm_core
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_core fern_feature_core fern_core)
ENDFOREACH()
