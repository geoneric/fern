#pragma once
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle if_                 (MaskedRasterHandle const& condition,
                                        MaskedRasterHandle const& true_value,
                                        MaskedRasterHandle const& false_value);

MaskedRasterHandle if_                 (MaskedRasterHandle const& condition,
                                        int64_t true_value,
                                        MaskedRasterHandle const& false_value);

MaskedRasterHandle if_                 (MaskedRasterHandle const& condition,
                                        MaskedRasterHandle const& true_value,
                                        int64_t false_value);

MaskedRasterHandle if_                 (MaskedRasterHandle const& condition,
                                        double true_value,
                                        MaskedRasterHandle const& false_value);

MaskedRasterHandle if_                 (MaskedRasterHandle const& condition,
                                        MaskedRasterHandle const& true_value,
                                        double false_value);

} // namespace python
} // namespace fern
