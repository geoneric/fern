configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/conf.py
)
configure_file(
    # Don't name the new file Makefile, as it conflicts with the CMake generated
    # file.
    ${CMAKE_CURRENT_SOURCE_DIR}/Makefile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Makefile-sphinx
)

foreach(NAME ${SPHINX_SOURCES})
    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${NAME})
    set(COPIED_SPHINX_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/${NAME})
    add_custom_command(
        OUTPUT ${COPIED_SPHINX_SOURCE}
        COMMAND ${CMAKE_COMMAND} -E copy ${SPHINX_SOURCE}
            ${COPIED_SPHINX_SOURCE}
        DEPENDS ${SPHINX_SOURCE}
    )
    list(APPEND COPIED_SPHINX_SOURCES ${COPIED_SPHINX_SOURCE})
endforeach()

foreach(NAME _build _static _templates)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}
        COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/${NAME}
    )
endforeach()

set(SPHINX_SPHINXOPTS "-q -W")
# set(SPHINX_PAPER a4)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_build/html/index.html
    COMMAND ${CMAKE_MAKE_PROGRAM} SPHINXOPTS=${SPHINX_SPHINXOPTS}
        -C ${CMAKE_CURRENT_BINARY_DIR}
        -f Makefile-sphinx html
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/conf.py
        ${CMAKE_CURRENT_BINARY_DIR}/Makefile-sphinx
        ${CMAKE_CURRENT_BINARY_DIR}/_build
        ${CMAKE_CURRENT_BINARY_DIR}/_static
        ${CMAKE_CURRENT_BINARY_DIR}/_templates
        ${COPIED_SPHINX_SOURCES}
)

add_custom_target(${SPHINX_TARGET} ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/_build/html/index.html
)
