SET(TEST_NAMES
    result_type_test
    result_types_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(operation_core_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(operation_core_${NAME}
        geoneric_operation_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME operation_core_${NAME} COMMAND operation_core_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
