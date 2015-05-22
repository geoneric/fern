set(TEST_NAMES
    python_test
)
copy_python_unit_test_modules(
    TARGET fern_feature_test_modules)
add_unit_tests(
    SCOPE python_extension_feature
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
    LINK_LIBRARIES
        fern_core
        ${PYTHON_LIBRARIES}
    DEPENDENCIES
        fern_feature_test_modules
        # The tests load the Python extension at runtime. Make sure it is
        # up to date.
        _fern_feature
)
