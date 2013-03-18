SET(TEST_NAMES
    algebra_parser_test
    annotate_visitor_test
    assignment_vertex_test
    definition_test
    dot_visitor_test
    expression_vertex_test
    function_vertex_test
    identify_visitor_test
    if_vertex_test
    name_vertex_test
    number_vertex_test
    operator_vertex_test
    # optimize_visitor_test
    script_vertex_test
    script_visitor_test
    statement_vertex_test
    string_vertex_test
    symbol_table_test
    syntax_vertex_test
    thread_visitor_test
    validate_visitor_test
    visitor_test
    while_vertex_test
    xml_parser_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(language_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(language_${NAME}
        ranally_interpreter
        ranally_language
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME language_${NAME} COMMAND language_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
