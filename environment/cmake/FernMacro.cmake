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

    # Evaluate the variable and include target if variable is defined and
    # evaluates to TRUE.
    if(${VARIABLE_NAME})
        add_subdirectory(${DIRECTORY_NAME})
    endif()
endmacro()
