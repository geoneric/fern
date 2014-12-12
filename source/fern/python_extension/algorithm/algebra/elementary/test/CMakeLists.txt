SET(TEST_NAMES
    python_test
)
COPY_PYTHON_UNIT_TEST_MODULES(
    TARGET fern_algorithm_algebra_elementary_test_modules)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE python_extension_algorithm_algebra_elementary
        NAME ${NAME}
        LINK_LIBRARIES ${PYTHON_LIBRARIES} stdc++
        DEPENDENCIES _fern_feature _fern_algorithm_algebra_elementary)
ENDFOREACH()
ADD_DEPENDENCIES(python_extension_algorithm_algebra_elementary_python_test
    fern_algorithm_algebra_elementary_test_modules fern_test_modules)
