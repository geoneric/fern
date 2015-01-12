SET(TEST_NAMES
    atan_test
    cos_test
    sin_test
    tan_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE algorithm_trigonometry
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy fern_core)
ENDFOREACH()
