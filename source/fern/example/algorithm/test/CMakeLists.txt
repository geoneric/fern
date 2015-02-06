set(TEST_NAMES
    raster_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE example_algorithm
        NAME ${NAME})
endforeach()
