include(CMakeFindDependencyMacro)
find_dependency(Boost
    COMPONENTS
        filesystem
        system
        timer
)
include("${CMAKE_CURRENT_LIST_DIR}/fern_targets.cmake")
