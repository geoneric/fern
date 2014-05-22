SET(TEST_NAMES
    result_type_test
    result_value_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_algebra_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_algebra_${NAME}
        # fern_algorithm_core
        # fern_feature_core  # fern::extents
        fern_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(
        NAME algorithm_algebra_${NAME}
        COMMAND algorithm_algebra_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
