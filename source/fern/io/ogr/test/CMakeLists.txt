set(TEST_NAMES
    ogr_dataset_driver_test
)

macro(ADD_UNIT_TEST NAME)
    add_executable(io_ogr_${NAME} ${NAME})
    target_link_libraries(io_ogr_${NAME}
        fern_io_ogr
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    add_test(NAME io_ogr_${NAME} COMMAND io_ogr_${NAME})
endmacro()

foreach(NAME ${TEST_NAMES})
    add_unit_test(${NAME})
endforeach()
