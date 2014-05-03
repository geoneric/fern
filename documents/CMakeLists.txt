IF(NOT FERN_ALGORITHM_ONLY)
    SET(LATEX_COMPILER_FLAGS
        # "--shell-escape
        "-interaction=nonstopmode --enable-write18"
        # CACHE STRING "Flags passed to latex."
    )
    INCLUDE(UseLATEX)


    ADD_SUBDIRECTORY(texts)
ENDIF()
