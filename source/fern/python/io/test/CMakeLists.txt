set(TEST_NAMES
    python_test
)
copy_python_unit_test_modules(
    TARGET fern_io_test_modules)
add_unit_tests(
    SCOPE python_extension_io
    NAMES ${TEST_NAMES}
    LINK_LIBRARIES
        fern_core
        ${PYTHON_LIBRARIES}
    DEPENDENCIES
        fern_io_test_modules
        fern_test_modules
        _fern_io
        _fern_feature
)
