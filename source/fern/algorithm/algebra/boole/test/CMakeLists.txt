set(TEST_NAMES
    and_test
    defined_test
    not_test
    or_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_algebra_boole
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy)
endforeach()
