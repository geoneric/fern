SET(TEST_NAMES
    raster_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE example_algorithm
        NAME ${NAME})
ENDFOREACH()
