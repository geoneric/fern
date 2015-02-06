set(TEST_NAMES
    operation_xml_parser_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(operation_xml_${NAME} ${NAME})
    target_link_libraries(operation_xml_${NAME}
        fern_operation_xml
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME operation_xml_${NAME} COMMAND operation_xml_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
