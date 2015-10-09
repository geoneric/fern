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
