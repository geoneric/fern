#pragma once
#include "fern/core/string.h"


namespace fern {
namespace python {

void               raise_runtime_error (String const& message);

void               raise_unsupported_argument_type_exception(
                                        String const& type_represenation);

} // namespace python
} // namespace fern
