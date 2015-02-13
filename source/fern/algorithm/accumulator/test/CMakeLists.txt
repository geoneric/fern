set(TEST_NAMES
    count_test
    max_test
    min_test
    mean_test
    sum_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_accumulator
        NAME ${NAME})
endforeach()
