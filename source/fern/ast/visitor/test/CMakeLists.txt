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

macro(ADD_UNIT_TEST NAME)
    add_executable(ast_visitor_${NAME} ${NAME})
    target_link_libraries(ast_visitor_${NAME}
        fern_ast_visitor
        fern_script
        fern_ast_xml
        fern_operation_std
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME ast_visitor_${NAME} COMMAND ast_visitor_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
