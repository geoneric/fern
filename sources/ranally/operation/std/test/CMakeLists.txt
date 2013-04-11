SET(TEST_NAMES
    print_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(operation_std_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(operation_std_${NAME}
        ranally_operation_std
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME operation_std_${NAME} COMMAND operation_std_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
