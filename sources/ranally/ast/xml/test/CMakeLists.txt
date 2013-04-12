SET(TEST_NAMES
    xml_parser_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(ast_xml_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(ast_xml_${NAME}
        ranally_ast_xml
        ranally_script
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME ast_xml_${NAME} COMMAND ast_xml_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
