set(TEST_NAMES
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
add_unit_tests(
    SCOPE language_ast_visitor
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_language_ast_visitor
        fern_language_script
        fern_language_ast_xml
        fern_language_operation_std)
