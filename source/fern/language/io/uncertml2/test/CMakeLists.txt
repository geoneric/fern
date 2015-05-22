set(TEST_NAMES
    parser_test
)
add_unit_tests(
    SCOPE language_io_uncertml2
    NAMES ${TEST_NAMES}
    OBJECT_LIBRARIES
        $<TARGET_OBJECTS:fern_language_io_uncertml2_objects>
        $<TARGET_OBJECTS:fern_language_uncertainty_objects>
    LINK_LIBRARIES
        fern_core
        ${EXPAT_LIBRARY}
)
