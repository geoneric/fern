set(TEST_NAMES
    python_test
)
copy_python_unit_test_modules(
    TARGET fern_algorithm_space_focal_test_modules)
add_unit_tests(
    SCOPE python_extension_algorithm_space_focal
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        ${PYTHON_LIBRARIES}
    DEPENDENCIES
        _fern_feature
        _fern_algorithm_space_focal
)
add_dependencies(python_extension_algorithm_space_focal_python_test
    fern_algorithm_space_focal_test_modules
    fern_test_modules
)
