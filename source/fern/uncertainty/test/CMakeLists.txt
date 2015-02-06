set(TEST_NAMES
    normal_distribution_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(uncertainty_${NAME} ${NAME})
    target_link_libraries(uncertainty_${NAME}
        fern_uncertainty
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME uncertainty_${NAME} COMMAND uncertainty_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
endforeach()
