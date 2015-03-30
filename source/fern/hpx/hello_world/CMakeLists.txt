add_executable(hello_world
    hello_world)
target_link_libraries(hello_world
    iostreams_component
    hpx_init
    hpx
    ${Boost_serialization_LIBRARY}
)
