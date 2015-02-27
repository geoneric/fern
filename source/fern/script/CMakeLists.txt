if(FERN_TEST)
    add_subdirectory(test)
endif()

set(SOURCES
    algebra_parser
)
add_library_and_object_library(fern_script ${SOURCES})
target_link_libraries(fern_script
    fern_python
    fern_core
    ${PYTHON_LIBRARIES}
)
