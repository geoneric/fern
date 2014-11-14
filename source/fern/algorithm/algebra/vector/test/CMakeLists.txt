SET(TEST_NAMES
    gradient_test
    laplacian_test
    lax_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(algorithm_algebra_vector ${NAME}
        fern_algorithm_core fern_feature_core)
ENDFOREACH()
