SET(TEST_NAMES
    equal_test
    plus_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_algebra_elementary_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_algebra_elementary_${NAME}
        fern_core
        fern_algorithm_core
        fern_feature_core  # fern::extents
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(
        NAME algorithm_algebra_elementary_${NAME}
        COMMAND algorithm_algebra_elementary_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
