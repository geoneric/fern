SET(TEST_NAMES
  convolve_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_convolution_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_convolution_${NAME}
        fern_feature_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(
        NAME algorithm_convolution_${NAME}
        COMMAND algorithm_convolution_${NAME}
    )
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
