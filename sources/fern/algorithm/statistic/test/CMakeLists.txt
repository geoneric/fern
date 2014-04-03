SET(TEST_NAMES
    count_test
    sum_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_statistic_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_statistic_${NAME}
        fern_core
        fern_algorithm_core
        fern_feature_core  # fern::extents
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(
        NAME algorithm_statistic_${NAME}
        COMMAND algorithm_statistic_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
