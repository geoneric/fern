set(TEST_NAMES
    constant_test
    evaluate_visitor_test
    expression_test
    raster_test
    stream_visitor_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(expression_tree_${NAME} ${NAME})
    target_link_libraries(expression_tree_${NAME}
        # fern_expression_tree
        fern_feature_core  # fern::extents
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME expression_tree_${NAME} COMMAND expression_tree_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
