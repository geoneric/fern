SET(TEST_NAMES
    operation_xml_parser_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(operation_xml_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(operation_xml_${NAME}
        fern_operation_xml
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME operation_xml_${NAME} COMMAND operation_xml_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
