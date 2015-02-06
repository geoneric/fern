set(TEST_NAMES
    drivers_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(io_${NAME} ${NAME})
    target_link_libraries(io_${NAME}
        fern_io
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME io_${NAME} COMMAND io_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
