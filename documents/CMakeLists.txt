### IF(NOT FERN_ALGORITHM_ONLY)
###     SET(LATEX_COMPILER_FLAGS
###         # "--shell-escape
###         "-interaction=nonstopmode --enable-write18"
###         # CACHE STRING "Flags passed to latex."
###     )
###     INCLUDE(UseLATEX)
### ENDIF()

ADD_SUBDIRECTORY(manual)
ADD_SUBDIRECTORY(texts)
