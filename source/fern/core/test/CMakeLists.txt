add_unit_tests(
    SCOPE core
    NAMES
        base_class_test.cc
        data_name_test.cc
        data_types_test.cc
        flag_collection_test.cc
        path_test.cc
        point_test.cc
        scope_exit_test.cc
        scope_test.cc
        stack_test.cc
        std_test.cc
        # stopwatch_test.cc
        string_test.cc
        type_traits_test.cc
        typelist_test.cc
        value_types_test.cc
    LINK_LIBRARIES
        fern::core
)
