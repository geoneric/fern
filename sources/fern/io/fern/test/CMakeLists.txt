SET(TEST_NAMES
    fern_dataset_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(io_fern_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(io_fern_${NAME}
        fern_io_geoneric
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME io_fern_${NAME} COMMAND io_fern_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
