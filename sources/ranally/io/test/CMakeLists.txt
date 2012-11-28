SET(TEST_NAMES
    hdf5_dataset_driver_test
    hdf5_dataset_test
    ogr_dataset_driver_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(${NAME}
        ranally_io
    )
    ADD_TEST(NAME ${NAME} COMMAND ${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
