set(TEST_NAMES
    algebra_parser_test
)
add_unit_tests(
    SCOPE language_script
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_language_python_objects>
        $<TARGET_OBJECTS:fern_language_script_objects>
    LINK_LIBRARIES
        fern_core
        ${PYTHON_LIBRARIES}
)
