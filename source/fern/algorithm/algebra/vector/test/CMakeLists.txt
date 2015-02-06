set(TEST_NAMES
    gradient_test
    laplacian_test
    lax_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE algorithm_algebra_vector
        NAME ${NAME}
        LINK_LIBRARIES fern_algorithm_policy fern_algorithm_core
            fern_feature_core)
endforeach()
