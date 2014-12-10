SET(TEST_NAMES
    python_test
)
COPY_PYTHON_UNIT_TEST_MODULES(
    TARGET fern_io_test_modules)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE python_extension_io
        NAME ${NAME}
        LINK_LIBRARIES fern_core ${PYTHON_LIBRARIES} stdc++
        DEPENDENCIES _fern_io _fern_feature)
ENDFOREACH()
ADD_DEPENDENCIES(python_extension_io_python_test
    fern_io_test_modules fern_test_modules)
