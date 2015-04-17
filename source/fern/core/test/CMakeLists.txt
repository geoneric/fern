set(TEST_NAMES
    base_class_test
    data_types_test
    flag_collection_test
    point_test
    scope_exit_test
    scope_test
    stack_test
    std_test
    stopwatch_test
    string_test
    symbol_table_test
    type_traits_test
    typelist_test
    value_types_test
)
if(FERN_COMPILER_HAS_REGEX)
    set(TEST_NAMES
        ${TEST_NAMES}
        data_name_test
        path_test
    )
endif()

foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE core
        NAME ${NAME}
        LINK_LIBRARIES fern_core)
endforeach()
