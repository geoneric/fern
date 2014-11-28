SET(TEST_NAMES
    add_test
    python_test
)
COPY_PYTHON_UNIT_TEST_MODULES(
    TARGET fern_algorithm_gdal_test_modules)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE python_extension_algorithm_gdal
        NAME ${NAME}
        LINK_LIBRARIES fern_core ${PYTHON_LIBRARIES} stdc++
        # The tests load the Python extension at runtime. Make sure it is
        # up to date.
        DEPENDENCIES fern_algorithm_gdal)
ENDFOREACH()
ADD_DEPENDENCIES(python_extension_algorithm_gdal_python_test
    fern_algorithm_gdal_test_modules)
