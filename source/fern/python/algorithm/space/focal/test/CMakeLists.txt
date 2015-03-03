set(TEST_NAMES
    python_test
)
copy_python_unit_test_modules(
    TARGET fern_algorithm_space_focal_test_modules)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE python_extension_algorithm_space_focal
        NAME ${NAME}
        LINK_LIBRARIES ${PYTHON_LIBRARIES} stdc++
        DEPENDENCIES _fern_feature _fern_algorithm_space_focal)
endforeach()
add_dependencies(python_extension_algorithm_space_focal_python_test
    fern_algorithm_space_focal_test_modules fern_test_modules)
