set(TEST_NAMES
    xml_parser_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(ast_xml_${NAME} ${NAME})
    target_link_libraries(ast_xml_${NAME}
        fern_ast_xml
        fern_script
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME ast_xml_${NAME} COMMAND ast_xml_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
