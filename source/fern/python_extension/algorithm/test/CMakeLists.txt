SET(TEST_NAMES
    python_test
)
COPY_PYTHON_UNIT_TEST_MODULES(
    TARGET fern_algorithm_test_modules)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE python_extension_algorithm
        NAME ${NAME}
        LINK_LIBRARIES fern_core ${PYTHON_LIBRARIES} stdc++
        # The tests load the Python extension at runtime. Make sure it is
        # up to date.
        DEPENDENCIES _fern_feature _fern_algorithm)
ENDFOREACH()
ADD_DEPENDENCIES(python_extension_algorithm_python_test
    fern_algorithm_test_modules)
