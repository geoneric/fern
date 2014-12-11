#pragma once
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle greater             (MaskedRasterHandle const& raster1,
                                        MaskedRasterHandle const& raster2);

MaskedRasterHandle greater             (int64_t value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle greater             (MaskedRasterHandle const& raster,
                                        int64_t value);

MaskedRasterHandle greater             (double value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle greater             (MaskedRasterHandle const& raster,
                                        double value);

} // namespace python
} // namespace fern
