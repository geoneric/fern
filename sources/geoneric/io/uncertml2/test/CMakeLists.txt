SET(TEST_NAMES
    parser_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(io_uncertml2_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(io_uncertml2_${NAME}
        geoneric_io_uncertml2
        geoneric_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME io_uncertml2_${NAME} COMMAND io_uncertml2_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
