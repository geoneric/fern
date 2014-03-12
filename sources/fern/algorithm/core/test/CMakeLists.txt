SET(TEST_NAMES
    index_range_test
    index_ranges_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_core_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_core_${NAME}
        fern_algorithm_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME algorithm_core_${NAME} COMMAND algorithm_core_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
