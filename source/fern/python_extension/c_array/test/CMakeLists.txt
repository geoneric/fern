SET(TEST_NAMES
    algorithm_c_array_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_c_array_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_c_array_${NAME}
        fern_algorithm_core
        fern_feature_core  # fern::extents
        # fern_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME algorithm_c_array_${NAME} COMMAND algorithm_c_array_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
