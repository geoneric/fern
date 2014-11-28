#pragma once
#include "fern/core/types.h"


namespace fern {
namespace python {
namespace core {

int64_t            add              (int64_t value1,
                                     int64_t value2);

float64_t          add              (float64_t value1,
                                     float64_t value2);

float64_t          add              (int64_t value1,
                                     float64_t value2);

} // namespace core
} // namespace python
} // namespace fern
