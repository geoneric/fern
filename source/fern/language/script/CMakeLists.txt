add_test_conditionally(test)

set(SOURCES
    algebra_parser
)
add_library_and_object_library(fern_language_script ${SOURCES})
target_link_libraries(fern_language_script
    fern_language_python
    fern_core
    ${PYTHON_LIBRARIES}
)
