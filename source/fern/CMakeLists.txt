CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/configure.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/configure.h
)

ADD_TARGET_CONDITIONALLY(algorithm algorithm)
ADD_TARGET_CONDITIONALLY(algorithm_python_extension python_extension)
ADD_TARGET_CONDITIONALLY(ast ast)
ADD_TARGET_CONDITIONALLY(back_end back_end)
ADD_TARGET_CONDITIONALLY(command command)
ADD_TARGET_CONDITIONALLY(compiler compiler)
ADD_TARGET_CONDITIONALLY(core core)
ADD_TARGET_CONDITIONALLY(example example)
ADD_TARGET_CONDITIONALLY(expression_tree expression_tree)
ADD_TARGET_CONDITIONALLY(feature feature)
ADD_TARGET_CONDITIONALLY(interpreter interpreter)
ADD_TARGET_CONDITIONALLY(io io)
ADD_TARGET_CONDITIONALLY(operation operation)
ADD_TARGET_CONDITIONALLY(python python)
ADD_TARGET_CONDITIONALLY(script script)
ADD_TARGET_CONDITIONALLY(uncertainty uncertainty)

IF(FERN_ALGORITHM)
    # Install all headers for now. Otherwise we have to keep tweaking the
    # install rules.
    INSTALL(
        DIRECTORY algorithm core
        DESTINATION include/fern
        FILES_MATCHING PATTERN "*.h"
    )
    INSTALL(
        DIRECTORY example
        DESTINATION doc
        FILES_MATCHING PATTERN "*.h"
        FILES_MATCHING PATTERN "*.cc"
    )
ENDIF()

### ELSE()
###     # set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols")
###     # For executables, you can use:
###     # ADD_EXECUTABLE(NAME_OF_EXECUTABLE $ $)
###     # SET(LINK_FLAGS ${LINK_FLAGS} "-Wl,-whole-archive")
###     # TARGET_LINK_LIBRARIES(NAME_OF_EXECUTABLE ${PROJECT_NAME})
### 
###     # Aggregate object libraries into a single shared library. This library is
###     # linked against by compiled modules.
###     ADD_LIBRARY(fernlib SHARED
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
###     TARGET_LINK_LIBRARIES(fernlib
###         ${FERN_EXTERNAL_LIBRARIES}
###     )
###     INSTALL(
###         TARGETS fernlib
###         EXPORT fern_targets
###         RUNTIME DESTINATION bin
###         LIBRARY DESTINATION lib
###     )
### 
###     # Install all header for now. Otherwise we have to keep tweaking the
###     # install rules.
###     INSTALL(
###         DIRECTORY .
###         DESTINATION include/fern
###         FILES_MATCHING PATTERN "*.h"
###     )
### ENDIF()
