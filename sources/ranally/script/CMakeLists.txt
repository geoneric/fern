ADD_SUBDIRECTORY(test)

SET(SOURCES
    algebra_parser
)
ADD_LIBRARY(ranally_script
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_script
    ranally_python
    ranally_core
    ${PYTHON_LIBRARIES}
)
