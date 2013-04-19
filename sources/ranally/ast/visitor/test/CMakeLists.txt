SET(TEST_NAMES
    annotate_visitor_test
    dot_visitor_test
    identify_visitor_test
    # This introduces a syclic dependency. Shouldn't depend on interpreter.
    # Or move to higher level place.
    # optimize_visitor_test
    script_visitor_test
    thread_visitor_test
    validate_visitor_test
    visitor_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(ast_visitor_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(ast_visitor_${NAME}
        ranally_ast_visitor
        ranally_script
        ranally_ast_xml
        ranally_operation_std
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME ast_visitor_${NAME} COMMAND ast_visitor_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
