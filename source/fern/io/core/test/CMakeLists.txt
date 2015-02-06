set(TEST_NAMES
    file_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(io_core_${NAME} ${NAME})
    target_link_libraries(io_core_${NAME}
        fern_io_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME io_core_${NAME} COMMAND io_core_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
