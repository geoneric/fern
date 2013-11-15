SET(TEST_NAMES
    annotate_visitor_test
    ast_visitor_test
    dot_visitor_test
    identify_visitor_test
    io_visitor_test
    module_visitor_test
    # This introduces a syclic dependency. Shouldn't depend on interpreter.
    # Or move to higher level place.
    # optimize_visitor_test
    thread_visitor_test
    validate_visitor_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(ast_visitor_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(ast_visitor_${NAME}
        geoneric_ast_visitor
        geoneric_script
        geoneric_ast_xml
        geoneric_operation_std
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME ast_visitor_${NAME} COMMAND ast_visitor_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
