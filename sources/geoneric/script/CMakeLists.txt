ADD_SUBDIRECTORY(test)

SET(SOURCES
    algebra_parser
)
ADD_LIBRARY(geoneric_script
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_script
    geoneric_python
    geoneric_core
    ${PYTHON_LIBRARIES}
)
