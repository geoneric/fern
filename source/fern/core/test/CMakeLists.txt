set(TEST_NAMES
    base_class_test
    data_name_test
    data_types_test
    flag_collection_test
    path_test
    point_test
    scope_exit_test
    scope_test
    stack_test
    std_test
    stopwatch_test
    string_test
    type_traits_test
    typelist_test
    value_types_test
)
add_unit_tests(
    SCOPE core
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_core
)
