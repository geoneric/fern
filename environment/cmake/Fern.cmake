# https://github.com/geoneric/peacock/blob/master/cmake/PeacockPlatform.cmake
include(PeacockPlatform) # This one first. Other modules use the variables.

if(WIN32)
    set(CMAKE_DEBUG_POSTFIX "d")
endif()

include(FernCompiler)  # This one first. Configuration uses the compiler.
include(FernConfiguration)
include(FernExternal)
include(FernMacros)
include(CppCheck)

if(FERN_TEST)
    enable_testing()
endif()
# Doesn't work: set(ENV{CTEST_OUTPUT_ON_FAILURE} 1)
# set(BOOST_TEST_RUNTIME_PARAMETERS "--log_level all")

force_out_of_tree_build()
