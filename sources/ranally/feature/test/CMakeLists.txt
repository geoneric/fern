SET(TEST_NAMES
    feature_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(feature_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(feature_${NAME}
        ranally_feature
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME feature_${NAME} COMMAND feature_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
