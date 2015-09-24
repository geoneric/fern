set(TEST_NAMES
    python_test
)
copy_python_unit_test_modules(
    TARGET fern_algorithm_test_modules)
add_unit_tests(
    SCOPE python_extension_algorithm
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        ${PYTHON_LIBRARIES}
    DEPENDENCIES
        _fern_feature
        _fern_algorithm
)
add_dependencies(python_extension_algorithm_python_test
    fern_algorithm_test_modules
    fern_test_modules
)
