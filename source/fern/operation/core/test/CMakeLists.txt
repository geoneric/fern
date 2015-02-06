set(TEST_NAMES
    expression_type_test
    expression_types_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(operation_core_${NAME} ${NAME})
    target_link_libraries(operation_core_${NAME}
        fern_operation_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME operation_core_${NAME} COMMAND operation_core_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
