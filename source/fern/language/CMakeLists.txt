add_target_conditionally(language_ast ast)
add_target_conditionally(language_back_end back_end)
add_target_conditionally(language_command command)
add_target_conditionally(language_compiler compiler)
add_target_conditionally(language_feature feature)
add_target_conditionally(language_interpreter interpreter)
add_target_conditionally(language_io io)
add_target_conditionally(language_operation operation)
add_target_conditionally(language_python python)
add_target_conditionally(language_script script)
add_target_conditionally(language_uncertainty uncertainty)


# set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols")
# For executables, you can use:
# ADD_EXECUTABLE(NAME_OF_EXECUTABLE $ $)
# SET(LINK_FLAGS ${LINK_FLAGS} "-Wl,-whole-archive")
# TARGET_LINK_LIBRARIES(NAME_OF_EXECUTABLE ${PROJECT_NAME})

# Aggregate object libraries into a single shared library. This library is
# linked against by compiled modules.
add_library(fern_language SHARED
    $<TARGET_OBJECTS:fern_language_compiler_objects>
    $<TARGET_OBJECTS:fern_language_ast_visitor_objects>
    $<TARGET_OBJECTS:fern_language_ast_core_objects>
    $<TARGET_OBJECTS:fern_language_ast_xml_objects>
    $<TARGET_OBJECTS:fern_language_feature_core_objects>
    $<TARGET_OBJECTS:fern_language_feature_visitor_objects>
    $<TARGET_OBJECTS:fern_language_interpreter_objects>
    $<TARGET_OBJECTS:fern_language_script_objects>
    $<TARGET_OBJECTS:fern_language_io_core_objects>
    $<TARGET_OBJECTS:fern_language_io_fern_objects>
    $<TARGET_OBJECTS:fern_language_io_gdal_objects>
    $<TARGET_OBJECTS:fern_language_io_objects>
    $<TARGET_OBJECTS:fern_language_operation_core_objects>
    $<TARGET_OBJECTS:fern_language_operation_std_objects>
    $<TARGET_OBJECTS:fern_language_python_objects>
    $<TARGET_OBJECTS:fern_feature_core_objects>
    $<TARGET_OBJECTS:fern_core_objects>
)
target_link_libraries(fern_language
    ${FERN_EXTERNAL_LIBRARIES}
)
install(
    TARGETS fern_language
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# # Install all header for now. Otherwise we have to keep tweaking the
# # install rules.
# install(
#     DIRECTORY .
#     DESTINATION include/fern
#     FILES_MATCHING PATTERN "*.h"
# )
