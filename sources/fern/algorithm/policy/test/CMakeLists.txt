SET(TEST_NAMES
    collect_no_data_indices_test
    mark_no_data_by_value_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_policy_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_policy_${NAME}
        # fern_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME algorithm_policy_${NAME} COMMAND algorithm_policy_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
