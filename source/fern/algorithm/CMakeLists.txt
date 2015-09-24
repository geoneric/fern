add_subdirectory(algebra)
add_subdirectory(accumulator)
add_subdirectory(convolution)
add_subdirectory(core)
add_subdirectory(policy)
add_subdirectory(space)
add_subdirectory(statistic)
add_test_conditionally(test)
add_subdirectory(trigonometry)


get_property(FERN_ALGORITHM_OBJECT_LIBRARIES GLOBAL
    PROPERTY FERN_ALGORITHM_OBJECT_LIBRARIES)


if(FERN_ALGORITHM_DYN_LINK)
    add_library(fern_algorithm SHARED
        ${FERN_ALGORITHM_OBJECT_LIBRARIES}
    )
else()
    add_library(fern_algorithm STATIC
        ${FERN_ALGORITHM_OBJECT_LIBRARIES}
    )
endif()
set_target_properties(fern_algorithm
    PROPERTIES
        COMPILE_FLAGS "-DFERN_ALGORITHM_SOURCE"
)
target_link_libraries(fern_algorithm
    fern_core
)
install(
    TARGETS fern_algorithm
    EXPORT fern_targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
