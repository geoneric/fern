### if(NOT FERN_ALGORITHM_ONLY)
###     set(LATEX_COMPILER_FLAGS
###         # "--shell-escape
###         "-interaction=nonstopmode --enable-write18"
###         # CACHE STRING "Flags passed to latex."
###     )
###     include(UseLATEX)
### endif()

if(FERN_BUILD_DOCUMENTATION)
    add_subdirectory(manual)
endif()
add_subdirectory(text)
