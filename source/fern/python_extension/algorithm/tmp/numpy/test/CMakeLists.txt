SET(TEST_NAMES
    add_test
    python_test
)
COPY_PYTHON_UNIT_TEST_MODULES(
    TARGET fern_algorithm_numpy_test_modules)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE python_extension_algorithm_numpy
        NAME ${NAME}
        LINK_LIBRARIES fern_core ${PYTHON_LIBRARIES} stdc++
        # The tests load the Python extension at runtime. Make sure it is
        # up to date.
        DEPENDENCIES _fern_algorithm_numpy)
ENDFOREACH()
ADD_DEPENDENCIES(python_extension_algorithm_numpy_python_test
    fern_algorithm_numpy_test_modules)
