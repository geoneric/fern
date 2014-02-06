SET(TEST_NAMES
    constant_test
    evaluate_visitor_test
    expression_test
    raster_test
    stream_visitor_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(expression_tree_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(expression_tree_${NAME}
        # fern_expression_tree
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME expression_tree_${NAME} COMMAND expression_tree_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
