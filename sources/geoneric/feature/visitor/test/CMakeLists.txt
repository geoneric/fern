SET(TEST_NAMES
    attribute_type_visitor_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(feature_visitor_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(feature_visitor_${NAME}
        geoneric_feature_visitor
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME feature_visitor_${NAME} COMMAND feature_visitor_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
