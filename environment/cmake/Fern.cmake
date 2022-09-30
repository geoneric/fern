set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fern assumes signed integers wrap. AppleClang Release build don't do this.
# This options forces AppleClang to behave similar to other compilers.
add_compile_options($<$<CXX_COMPILER_ID:AppleClang>:-fwrapv>)

set(CMAKE_DEBUG_POSTFIX "d")

include(PeacockPlatform)
include(DevBaseCompiler)
include(FernConfiguration)
include(DevBaseExternal)
include(DevBaseMacro)
include(FernMacro)
include(CppCheck)

if(FERN_BUILD_TEST)
    enable_testing()
endif()
# Doesn't work: set(ENV{CTEST_OUTPUT_ON_FAILURE} 1)
# set(BOOST_TEST_RUNTIME_PARAMETERS "--log_level all")

force_out_of_tree_build()

set(FERN_DATA_DIR ${PROJECT_SOURCE_DIR}/data)

# TODO Print status information about the current build.
