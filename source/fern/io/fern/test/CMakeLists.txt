set(TEST_NAMES
    fern_dataset_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(io_fern_${NAME} ${NAME})
    target_link_libraries(io_fern_${NAME}
        fern_io_fern
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME io_fern_${NAME} COMMAND io_fern_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
