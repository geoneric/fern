// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
