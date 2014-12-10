SET(TEST_NAMES
    python_test
)
COPY_PYTHON_UNIT_TEST_MODULES(
    TARGET fern_feature_test_modules)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE python_extension_feature
        NAME ${NAME}
        LINK_LIBRARIES fern_core ${PYTHON_LIBRARIES} stdc++
        # The tests load the Python extension at runtime. Make sure it is
        # up to date.
        DEPENDENCIES _fern_feature)
ENDFOREACH()
ADD_DEPENDENCIES(python_extension_feature_python_test
    fern_feature_test_modules)
