add_executable(fern
    command
    compile_command
    describe_command
    execute_command
    import_command
    interpreter
    message
    fern
)
add_dependencies(fern
    Operation-xml.h
)
target_link_libraries(fern
    fernlib
    ${READLINE_LIBRARY}
    termcap
)
install(
    TARGETS fern
    EXPORT fern_targets
    RUNTIME DESTINATION bin
)
set(APPLICATIONS
    "\${CMAKE_INSTALL_PREFIX}/bin/fern"
)
