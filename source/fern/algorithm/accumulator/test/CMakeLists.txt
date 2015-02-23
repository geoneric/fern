set(TEST_NAMES
    count_test
    frequency_table_test
    max_test
    min_test
    mean_test
    median_test
    mode_test
    range_test
    standard_deviation_test
    sum_test
    variance_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_accumulator
        NAME ${NAME})
endforeach()
