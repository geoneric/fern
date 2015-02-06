set(TEST_NAMES
    abs_test
    operations_test
    print_test
)

macro(add_unit_test NAME)
    add_executable(operation_std_${NAME} ${NAME})
    target_link_libraries(operation_std_${NAME}
        fern_operation_std
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME operation_std_${NAME} COMMAND operation_std_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
