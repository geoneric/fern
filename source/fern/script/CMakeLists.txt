ADD_SUBDIRECTORY(test)

SET(SOURCES
    algebra_parser
)
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_script ${SOURCES})
TARGET_LINK_LIBRARIES(fern_script
    fern_python
    fern_core
    ${PYTHON_LIBRARIES}
)
