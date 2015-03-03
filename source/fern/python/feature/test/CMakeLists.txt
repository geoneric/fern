set(TEST_NAMES
    python_test
)
copy_python_unit_test_modules(
    TARGET fern_feature_test_modules)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE python_extension_feature
        NAME ${NAME}
        LINK_LIBRARIES fern_core ${PYTHON_LIBRARIES} stdc++
        # The tests load the Python extension at runtime. Make sure it is
        # up to date.
        DEPENDENCIES _fern_feature)
endforeach()
add_dependencies(python_extension_feature_python_test
    fern_feature_test_modules)
