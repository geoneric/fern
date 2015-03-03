set(TEST_NAMES
    add_test
    python_test
)
copy_python_unit_test_modules(
    TARGET fern_algorithm_gdal_test_modules)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE python_extension_algorithm_gdal
        NAME ${NAME}
        LINK_LIBRARIES fern_core ${PYTHON_LIBRARIES} stdc++
        # The tests load the Python extension at runtime. Make sure it is
        # up to date.
        DEPENDENCIES fern_algorithm_gdal)
endforeach()
add_dependencies(python_extension_algorithm_gdal_python_test
    fern_algorithm_gdal_test_modules)
