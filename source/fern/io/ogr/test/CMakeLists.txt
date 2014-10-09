SET(TEST_NAMES
    ogr_dataset_driver_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(io_ogr_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(io_ogr_${NAME}
        fern_io_ogr
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME io_ogr_${NAME} COMMAND io_ogr_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
