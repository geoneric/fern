SET(TEST_NAMES
  square_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_convolution_neighborhood_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_convolution_neighborhood_${NAME}
        fern_feature_core
        fern_algorithm_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(
        NAME algorithm_convolution_neighborhood_${NAME}
        COMMAND algorithm_convolution_neighborhood_${NAME}
    )
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
