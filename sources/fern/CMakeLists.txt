CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/configure.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/configure.h
)

ADD_SUBDIRECTORY(algorithm)
ADD_SUBDIRECTORY(ast)
ADD_SUBDIRECTORY(back_end)
ADD_SUBDIRECTORY(command)
ADD_SUBDIRECTORY(compiler)
ADD_SUBDIRECTORY(core)
ADD_SUBDIRECTORY(example)
ADD_SUBDIRECTORY(expression_tree)
ADD_SUBDIRECTORY(feature)
ADD_SUBDIRECTORY(interpreter)
ADD_SUBDIRECTORY(io)
ADD_SUBDIRECTORY(operation)
ADD_SUBDIRECTORY(python)
ADD_SUBDIRECTORY(script)
ADD_SUBDIRECTORY(uncertainty)

# set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--export-all-symbols")
# For executables, you can use:
# ADD_EXECUTABLE(NAME_OF_EXECUTABLE $ $)
# SET(LINK_FLAGS ${LINK_FLAGS} "-Wl,-whole-archive")
# TARGET_LINK_LIBRARIES(NAME_OF_EXECUTABLE ${PROJECT_NAME})

# Aggregate object libraries into a single shared library. This library is
# linked against by compiled modules.
ADD_LIBRARY(fernlib SHARED
    $<TARGET_OBJECTS:fern_compiler_objects>
    $<TARGET_OBJECTS:fern_ast_visitor_objects>
    $<TARGET_OBJECTS:fern_ast_core_objects>
    $<TARGET_OBJECTS:fern_ast_xml_objects>
    $<TARGET_OBJECTS:fern_feature_core_objects>
    $<TARGET_OBJECTS:fern_feature_visitor_objects>
    $<TARGET_OBJECTS:fern_interpreter_objects>
    $<TARGET_OBJECTS:fern_script_objects>
    $<TARGET_OBJECTS:fern_io_core_objects>
    $<TARGET_OBJECTS:fern_io_fern_objects>
    $<TARGET_OBJECTS:fern_io_gdal_objects>
    $<TARGET_OBJECTS:fern_io_objects>
    $<TARGET_OBJECTS:fern_operation_core_objects>
    $<TARGET_OBJECTS:fern_operation_std_objects>
    $<TARGET_OBJECTS:fern_python_objects>
    $<TARGET_OBJECTS:fern_core_objects>
)
TARGET_LINK_LIBRARIES(fernlib
    ${FERN_EXTERNAL_LIBRARIES}
)
INSTALL(
    TARGETS fernlib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Install all header for now. Otherwise we have to keep tweaking the install
# rules.
INSTALL(
    DIRECTORY .
    DESTINATION include/fern
    FILES_MATCHING PATTERN "*.h"
)
