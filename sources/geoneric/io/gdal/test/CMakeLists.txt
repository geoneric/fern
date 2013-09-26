SET(TEST_NAMES
    drivers_test
    gdal_dataset_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(io_gdal_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(io_gdal_${NAME}
        geoneric_io_gdal
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME io_gdal_${NAME} COMMAND io_gdal_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
