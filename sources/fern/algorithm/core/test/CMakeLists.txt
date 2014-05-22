SET(TEST_NAMES
    cast_test
    index_range_test
    index_ranges_test
    unary_operation_test
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
