set(TEST_NAMES
    gdal_dataset_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(io_gdal_${NAME} ${NAME})
    target_link_libraries(io_gdal_${NAME}
        fern_io_gdal
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME io_gdal_${NAME} COMMAND io_gdal_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
endforeach()
