set(TEST_NAMES
    algorithm_c_array_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(algorithm_c_array_${NAME} ${NAME})
    target_link_libraries(algorithm_c_array_${NAME}
        fern_algorithm_core
        fern_feature_core  # fern::extents
        # fern_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME algorithm_c_array_${NAME} COMMAND algorithm_c_array_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
