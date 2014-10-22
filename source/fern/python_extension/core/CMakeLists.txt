ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/swig_runtime.h
    COMMAND echo "\\#pragma once" > swig_runtime.h
    COMMAND echo "\\#pragma GCC system_header" >> swig_runtime.h
    COMMAND echo "/*! @ingroup MyGlobals" >> swig_runtime.h
    COMMAND echo " * @{ */" >> swig_runtime.h
    COMMAND ${SWIG_EXECUTABLE} -python -external-runtime swig_runtime.h.tmp
    COMMAND cat swig_runtime.h.tmp >> swig_runtime.h
    COMMAND rm swig_runtime.h.tmp
    COMMAND echo "/*! @} */" >> swig_runtime.h
)
ADD_CUSTOM_TARGET(swig_runtime DEPENDS swig_runtime.h)


SET(SOURCES
    error
    swig
)
ADD_LIBRARY(fern_python_extension_core ${SOURCES})
ADD_DEPENDENCIES(fern_python_extension_core swig_runtime)
# TARGET_LINK_LIBRARIES(fern_python_extension_core
# )
