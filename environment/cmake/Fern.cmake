# https://github.com/geoneric/peacock/blob/master/cmake/PeacockPlatform.cmake
INCLUDE(PeacockPlatform) # This one first. Other modules use the variables.

IF(WIN32)
    SET(CMAKE_DEBUG_POSTFIX "d")
ENDIF()

INCLUDE(FernCompiler)  # This one first. Configuration uses the compiler.
INCLUDE(FernConfiguration)
INCLUDE(FernExternal)
INCLUDE(FernMacros)
