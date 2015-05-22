add_target_conditionally(language_ast ast)
add_target_conditionally(language_back_end back_end)
add_target_conditionally(language_command command)
add_target_conditionally(language_compiler compiler)
add_target_conditionally(language_core core)
add_target_conditionally(language_feature feature)
add_target_conditionally(language_interpreter interpreter)
add_target_conditionally(language_io io)
add_target_conditionally(language_operation operation)
add_target_conditionally(language_python python)
add_target_conditionally(language_script script)
add_target_conditionally(language_uncertainty uncertainty)


get_property(FERN_LANGUAGE_OBJECT_LIBRARIES GLOBAL
    PROPERTY FERN_LANGUAGE_OBJECT_LIBRARIES)
add_library(fern_language SHARED
    ${FERN_LANGUAGE_OBJECT_LIBRARIES}
)
target_link_libraries(fern_language
    fern_feature
    ${EXPAT_LIBRARIES}
    ${GDAL_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${PYTHON_LIBRARIES}
)
install(
    TARGETS fern_language
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
