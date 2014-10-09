SET(TEST_NAMES
    binary_local_operation_test
    cast_test
    cover_test
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

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_core_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_core_${NAME}
        fern_algorithm_core
        fern_feature_core  # fern::extents
        fern_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME algorithm_core_${NAME} COMMAND algorithm_core_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
