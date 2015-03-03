#pragma once
#include "fern/core/string.h"


namespace fern {
namespace python {

void               raise_runtime_error (String const& message);

void               raise_unsupported_argument_type_exception(
                                        String const& type_represenation);

void               raise_unsupported_argument_type_exception(
                                        PyObject* object);

void               raise_unsupported_overload_exception(
                                        String const& type_representation);

void               raise_unsupported_overload_exception(
                                        String const& type_representation1,
                                        String const& type_representation2);

void               raise_unsupported_overload_exception(
                                        PyObject* object);

void               raise_unsupported_overload_exception(
                                        PyObject* object1,
                                        PyObject* object2);

} // namespace python
} // namespace fern
