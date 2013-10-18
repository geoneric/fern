SET(TEST_NAMES
    geoneric_dataset_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(io_geoneric_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(io_geoneric_${NAME}
        geoneric_io_geoneric
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME io_geoneric_${NAME} COMMAND io_geoneric_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
