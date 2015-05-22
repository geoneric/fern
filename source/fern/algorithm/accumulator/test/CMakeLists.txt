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
add_unit_tests(
    SCOPE algorithm_accumulator
    NAMES ${TEST_NAMES}
)
