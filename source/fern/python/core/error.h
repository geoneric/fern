// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
