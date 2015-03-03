#pragma once


#define INIT_PYTHON_MODULE(                                               \
    doc_string)                                                           \
bp::scope().attr("__doc__") = doc_string;                                 \
                                                                          \
bool const show_user_defined = true;                                      \
bool const show_py_signatures = true;                                     \
bool const show_cpp_signatures = false;                                   \
bp::docstring_options doc_options(show_user_defined, show_py_signatures,  \
    show_cpp_signatures);
