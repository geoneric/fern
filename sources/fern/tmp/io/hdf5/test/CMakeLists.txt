SET(TEST_NAMES
    hdf5_dataset_driver_test
    hdf5_dataset_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(io_hdf5_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(io_hdf5_${NAME}
        geoneric_io_hdf5
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME io_hdf5_${NAME} COMMAND io_hdf5_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
