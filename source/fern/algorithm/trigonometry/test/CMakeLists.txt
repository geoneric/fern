set(TEST_NAMES
    atan_test
    cos_test
    sin_test
    tan_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_trigonometry
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy fern_core)
endforeach()
