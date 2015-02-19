set(TEST_NAMES
    count_test
    histogram_test
    majority_test
    max_test
    min_test
    minority_test
    mean_test
    median_test
    sum_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_accumulator
        NAME ${NAME})
endforeach()
