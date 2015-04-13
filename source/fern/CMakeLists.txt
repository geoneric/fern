configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configure.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/configure.h
)

add_target_conditionally(algorithm algorithm)
add_target_conditionally(ast ast)
add_target_conditionally(back_end back_end)
add_target_conditionally(command command)
add_target_conditionally(compiler compiler)
add_target_conditionally(core core)
add_target_conditionally(example example)
add_target_conditionally(expression_tree expression_tree)
add_target_conditionally(feature feature)
add_target_conditionally(interpreter interpreter)
add_target_conditionally(io io)
add_target_conditionally(operation operation)
add_target_conditionally(python python)
add_target_conditionally(script script)
add_target_conditionally(uncertainty uncertainty)
# add_target_conditionally(hpx hpx)

if(FERN_BUILD_ALGORITHM)
    # Install all headers for now. Otherwise we have to keep tweaking the
    # install rules.
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/configure.h
        DESTINATION include/fern
        CONFIGURATIONS Release
    )
    install(
        DIRECTORY algorithm core
        DESTINATION include/fern
        CONFIGURATIONS Release
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "test" EXCLUDE
    )
    install(
        DIRECTORY example
        DESTINATION doc/fern
        CONFIGURATIONS Release
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.cc"
            PATTERN "test" EXCLUDE
    )
endif()

### else()
###     # set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols")
###     # For executables, you can use:
###     # ADD_EXECUTABLE(NAME_OF_EXECUTABLE $ $)
###     # SET(LINK_FLAGS ${LINK_FLAGS} "-Wl,-whole-archive")
###     # TARGET_LINK_LIBRARIES(NAME_OF_EXECUTABLE ${PROJECT_NAME})
### 
###     # Aggregate object libraries into a single shared library. This library is
###     # linked against by compiled modules.
###     add_library(fernlib SHARED
###         $<TARGET_OBJECTS:fern_compiler_objects>
###         $<TARGET_OBJECTS:fern_ast_visitor_objects>
###         $<TARGET_OBJECTS:fern_ast_core_objects>
###         $<TARGET_OBJECTS:fern_ast_xml_objects>
###         $<TARGET_OBJECTS:fern_feature_core_objects>
###         $<TARGET_OBJECTS:fern_feature_visitor_objects>
###         $<TARGET_OBJECTS:fern_interpreter_objects>
###         $<TARGET_OBJECTS:fern_script_objects>
###         $<TARGET_OBJECTS:fern_io_core_objects>
###         $<TARGET_OBJECTS:fern_io_fern_objects>
###         $<TARGET_OBJECTS:fern_io_gdal_objects>
###         $<TARGET_OBJECTS:fern_io_objects>
###         $<TARGET_OBJECTS:fern_operation_core_objects>
###         $<TARGET_OBJECTS:fern_operation_std_objects>
###         $<TARGET_OBJECTS:fern_python_objects>
###         $<TARGET_OBJECTS:fern_core_objects>
###     )
###     target_link_libraries(fernlib
###         ${FERN_EXTERNAL_LIBRARIES}
###     )
###     install(
###         TARGETS fernlib
###         EXPORT fern_targets
###         RUNTIME DESTINATION bin
###         LIBRARY DESTINATION lib
###     )
### 
###     # Install all header for now. Otherwise we have to keep tweaking the
###     # install rules.
###     install(
###         DIRECTORY .
###         DESTINATION include/fern
###         FILES_MATCHING PATTERN "*.h"
###     )
### endif()
