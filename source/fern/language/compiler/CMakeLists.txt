add_test_conditionally(test)

set(SOURCES
    compile_visitor
    compiler
    data_description
    module
    parse_command_line
)
add_object_library(
    TARGET fern_language_compiler_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
# target_link_libraries(fern_language_compiler
#     fern_language_interpreter
#     # ${Boost_PROGRAM_OPTIONS_LIBRARY}
# )
