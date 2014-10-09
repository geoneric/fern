SET(TEST_NAMES
    normal_distribution_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(uncertainty_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(uncertainty_${NAME}
        fern_uncertainty
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME uncertainty_${NAME} COMMAND uncertainty_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
