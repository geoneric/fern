# Targets can be added conditionally. When Fern components are selected during
# configuration, certain variables are set that tell us whether or not to
# include a target or group of targets. See FernConfiguration.cmake.
# TARGET_NAME   : Name of target, snake_case.
# DIRECTORY_NAME: Name of subdirectory containing the target.
macro(add_target_conditionally
        TARGET_NAME
        DIRECTORY_NAME)
    # Determine name of variable to check.
    string(TOUPPER ${TARGET_NAME} VARIABLE_NAME)
    set(VARIABLE_NAME "FERN_FERN_${VARIABLE_NAME}_REQUIRED")

    # Make sure it is defined.
    if(NOT DEFINED ${VARIABLE_NAME})
        message(SEND_ERROR "Variable ${VARIABLE_NAME} is not defined")
    endif()

    # Evaluate the variable and include target if result is true.
    if(${VARIABLE_NAME})
        add_subdirectory(${DIRECTORY_NAME})
    endif()
endmacro()


# Tests can be added conditionally. When the Fern build is configured, the
# FERN_BUILD_TEST variable is set to TRUE or FALSE. Depending on its setting
# tests are build or not. See FernConfiguration.cmake.
# DIRECTORY_NAME: Name of subdirectory containing the target.
function(add_test_conditionally
        DIRECTORY_NAME)
    if(FERN_BUILD_TEST)
        add_subdirectory(${DIRECTORY_NAME})
    endif()
endfunction()
