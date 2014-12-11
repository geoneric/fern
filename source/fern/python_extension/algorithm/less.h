#pragma once
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle less                (MaskedRasterHandle const& raster1,
                                        MaskedRasterHandle const& raster2);

MaskedRasterHandle less                (int64_t value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle less                (MaskedRasterHandle const& raster,
                                        int64_t value);

MaskedRasterHandle less                (double value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle less                (MaskedRasterHandle const& raster,
                                        double value);

} // namespace python
} // namespace fern
