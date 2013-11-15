SET(TEST_NAMES
    algebra_parser_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(script_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(script_${NAME}
        geoneric_script
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME script_${NAME} COMMAND script_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
