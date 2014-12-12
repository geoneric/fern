#pragma once
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle&
                   iadd                (MaskedRasterHandle& self,
                                        MaskedRasterHandle const& other);

MaskedRasterHandle add                 (MaskedRasterHandle const& raster1,
                                        MaskedRasterHandle const& raster2);

MaskedRasterHandle add                 (int64_t value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle add                 (MaskedRasterHandle const& raster,
                                        int64_t value);

MaskedRasterHandle add                 (double value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle add                 (MaskedRasterHandle const& raster,
                                        double value);

} // namespace python
} // namespace fern
