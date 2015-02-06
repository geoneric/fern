set(TEST_NAMES
    parser_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(io_uncertml2_${NAME} ${NAME})
    target_link_libraries(io_uncertml2_${NAME}
        fern_io_uncertml2
        fern_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME io_uncertml2_${NAME} COMMAND io_uncertml2_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
