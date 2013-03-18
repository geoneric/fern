SET(TEST_NAMES
    data_types_test
    operation_xml_parser_test
    print_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(operation_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(operation_${NAME}
        ranally_operation
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME operation_${NAME} COMMAND operation_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
