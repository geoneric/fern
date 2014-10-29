IF(WIN32)
    SET(CMAKE_DEBUG_POSTFIX "d")
ENDIF()

# https://github.com/geoneric/peacock/blob/master/cmake/PeacockPlatform.cmake
INCLUDE(PeacockPlatform)

INCLUDE(FernConfiguration)
INCLUDE(FernCompiler)
INCLUDE(FernExternal)
INCLUDE(FernMacros)
