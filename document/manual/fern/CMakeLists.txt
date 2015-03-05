include(SphinxDoc)

if(FERN_ALGORITHM)
    add_subdirectory(algorithm)
endif()

if(FERN_IO)
    add_subdirectory(io)
endif()

if(FERN_PYTHON)
    add_subdirectory(python)
endif()

file(GLOB_RECURSE SPHINX_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    *.rst
    *.cc
)

set(SPHINX_SOURCES
    ${SPHINX_SOURCES}
    ${FERN_ALGORITHM_SPHINX_SOURCES}
    ${FERN_IO_SPHINX_SOURCES}
    ${FERN_PYTHON_SPHINX_SOURCES}
)

# set(SPHINX_TARGET
#     fern_manual
# )

# include(SphinxDoc)

create_sphinx_docs(
    TARGET fern_manual
    SOURCES ${SPHINX_SOURCES}
)

if(FERN_PYTHON)
    document_python_api(
        TARGET python_api
        SOURCE_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern
        BINARY_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/api)
    add_dependencies(python_api fern_python_package)
    add_dependencies(fern_manual python_api)
endif()
