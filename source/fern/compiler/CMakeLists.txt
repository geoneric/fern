add_subdirectory(test)

set(SOURCES
    compile_visitor
    compiler
    data_description
    module
    parse_command_line
)
add_library_and_object_library(fern_compiler ${SOURCES})
target_link_libraries(fern_compiler
    fern_interpreter
    # ${Boost_PROGRAM_OPTIONS_LIBRARY}
)
