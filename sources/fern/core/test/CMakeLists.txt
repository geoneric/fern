SET(TEST_NAMES
    base_class_test
    data_name_test
    data_types_test
    flag_collection_test
    path_test
    scope_test
    stack_test
    std_test
    stopwatch_test
    string_test
    symbol_table_test
    typelist_test
    type_traits_test
    value_types_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(core_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(core_${NAME}
        fern_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME core_${NAME} COMMAND core_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
