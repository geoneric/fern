SET(TEST_NAMES
    atan_test
    cos_test
    sin_test
    tan_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(algorithm_trigonometry ${NAME} fern_core)
ENDFOREACH()
