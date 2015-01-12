SET(TEST_NAMES
    and_test
    defined_test
    not_test
    or_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE algorithm_algebra_boole
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy)
ENDFOREACH()
