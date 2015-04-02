### if(NOT FERN_ALGORITHM_ONLY)
###     set(LATEX_COMPILER_FLAGS
###         # "--shell-escape
###         "-interaction=nonstopmode --enable-write18"
###         # CACHE STRING "Flags passed to latex."
###     )
###     include(UseLATEX)
### endif()

add_subdirectory(manual)
add_subdirectory(text)
