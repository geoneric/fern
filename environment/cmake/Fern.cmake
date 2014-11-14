IF(WIN32)
    SET(CMAKE_DEBUG_POSTFIX "d")
ENDIF()

# https://github.com/geoneric/peacock/blob/master/cmake/PeacockPlatform.cmake
INCLUDE(PeacockPlatform)

INCLUDE(FernCompiler)  # This one first. Configuration uses compiler.
INCLUDE(FernConfiguration)
INCLUDE(FernExternal)
INCLUDE(FernMacros)
