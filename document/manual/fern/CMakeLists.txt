include(SphinxDoc)

if(FERN_ALGORITHM)
    add_subdirectory(algorithm)

    file(GLOB_RECURSE FERN_ALGORITHM_SPHINX_SOURCES
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        algorithm/*.rst
        algorithm/*.cc
    )

    list(APPEND FERN_SPHINX_TOC_MODULES algorithm/index)
endif()

if(FERN_IO)
    add_subdirectory(io)

    file(GLOB_RECURSE FERN_IO_SPHINX_SOURCES
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        io/*.rst
        io/*.cc
    )

    list(APPEND FERN_SPHINX_TOC_MODULES io/index)
endif()

if(FERN_PYTHON)
    add_subdirectory(python)

    file(GLOB_RECURSE FERN_PYTHON_SPHINX_SOURCES
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        python/*.rst
        python/*.cc
    )

    list(APPEND FERN_SPHINX_TOC_MODULES python/index)
endif()

file(GLOB SPHINX_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    *.rst
)

set(SPHINX_SOURCES
    ${SPHINX_SOURCES}
    ${FERN_ALGORITHM_SPHINX_SOURCES}
    ${FERN_IO_SPHINX_SOURCES}
    ${FERN_PYTHON_SPHINX_SOURCES}
)

# Hack to get the correct items in the TOC.
string(REPLACE ";" "\n   " FERN_SPHINX_TOC_MODULES "${FERN_SPHINX_TOC_MODULES}")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/index.rst.in
    ${CMAKE_CURRENT_BINARY_DIR}/index.rst
)

create_sphinx_docs(
    TARGET fern_manual
    SOURCES ${SPHINX_SOURCES}
)

if(FERN_PYTHON)
    document_python_api(
        TARGET python_api
        SOURCE_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern
        BINARY_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/api
        # TODO Doesn't work yet.
        # EXCLUDE_DIRECTORIES ${PROJECT_BINARY_DIR}/bin/fern/test
        #     ${PROJECT_BINARY_DIR}/bin/fern/test/*
        #     test
        #     test/*
    )
    add_dependencies(python_api fern_python_package)
    add_dependencies(fern_manual python_api)
endif()
