set(TEST_NAMES
    python_test
)
copy_python_unit_test_modules(
    TARGET fern_io_test_modules)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE python_extension_io
        NAME ${NAME}
        LINK_LIBRARIES fern_core ${PYTHON_LIBRARIES} stdc++
        DEPENDENCIES _fern_io _fern_feature)
endforeach()
add_dependencies(python_extension_io_python_test
    fern_io_test_modules fern_test_modules)
