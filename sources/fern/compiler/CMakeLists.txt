ADD_SUBDIRECTORY(test)

SET(SOURCES
    compile_visitor
    compiler
    data_description
    module
    parse_command_line
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_compiler ${SOURCES})
TARGET_LINK_LIBRARIES(fern_compiler
    fern_interpreter
    # ${Boost_PROGRAM_OPTIONS_LIBRARY}
)
