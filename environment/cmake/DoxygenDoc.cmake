SET(DOXYGEN_TEMPLATE "
    QUIET                   = YES
    WARNINGS                = NO  # Turn on when writing docs.
    WARN_IF_DOC_ERROR       = NO  # Idem.
    WARN_NO_PARAMDOC        = NO  # Idem.
    ALWAYS_DETAILED_SEC     = YES
    INLINE_INHERITED_MEMB   = YES
    INHERIT_DOCS            = YES
    EXTRACT_ALL             = YES
    EXTRACT_PRIVATE         = NO
    EXTRACT_STATIC          = YES
    SOURCE_BROWSER          = YES
    FILE_PATTERNS           = *.h *.hpp *.hxx *.c *.cc *.cpp *.cxx *.dox *.md
    EXCLUDE_PATTERNS        = *Test.h *Test.cc *_test.cc
    EXPAND_ONLY_PREDEF      = NO
    GENERATE_LATEX          = NO
    RECURSIVE               = YES
    SORT_MEMBER_DOCS        = NO
    USE_MATHJAX             = YES
    MATHJAX_EXTENSIONS      = TeX/AMSmath TeX/AMSsymbols
")

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
)

# ADD_CUSTOM_COMMAND(
#     OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
#     COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#     DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
# )

# This target is *always considered out of date*. No, that's annoying.
ADD_CUSTOM_TARGET(cpp_doc # ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile

    # DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
)

# ADD_CUSTOM_TARGET(cpp_doc
#     ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#     DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     COMMENT "Generating API documentation" VERBATIM
# )
