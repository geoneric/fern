configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configure.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/configure.h
)

if(FERN_BUILD_LANGUAGE)
    add_subdirectory(language)
endif()


add_target_conditionally(algorithm algorithm)
add_target_conditionally(core core)
add_target_conditionally(example example)
add_target_conditionally(expression_tree expression_tree)
add_target_conditionally(feature feature)
add_target_conditionally(io io)
add_target_conditionally(python python)
# add_target_conditionally(hpx hpx)


if(FERN_BUILD_ALGORITHM)
    # Install all headers for now. Otherwise we have to keep tweaking the
    # install rules.
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/configure.h
        DESTINATION include/fern
    )
    install(
        DIRECTORY algorithm core
        DESTINATION include/fern
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "test" EXCLUDE
    )
    install(
        DIRECTORY example
        DESTINATION doc/fern
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.cc"
            PATTERN "test" EXCLUDE
    )
endif()
