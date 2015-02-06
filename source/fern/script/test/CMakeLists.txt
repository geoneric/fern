set(TEST_NAMES
    algebra_parser_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(script_${NAME} ${NAME})
    target_link_libraries(script_${NAME}
        fern_script
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME script_${NAME} COMMAND script_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
