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
