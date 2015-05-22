add_custom_command(
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
add_custom_target(swig_runtime DEPENDS swig_runtime.h)


set(SOURCES
    error
    swig
)
add_library(fern_python_extension_core ${SOURCES})
add_dependencies(fern_python_extension_core swig_runtime)
