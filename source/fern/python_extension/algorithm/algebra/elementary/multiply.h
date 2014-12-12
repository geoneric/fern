#pragma once
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle&
                   imultiply           (MaskedRasterHandle& self,
                                        MaskedRasterHandle const& other);

MaskedRasterHandle multiply            (MaskedRasterHandle const& raster1,
                                        MaskedRasterHandle const& raster2);

MaskedRasterHandle multiply            (int64_t value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle multiply            (MaskedRasterHandle const& raster,
                                        int64_t value);

MaskedRasterHandle multiply            (double value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle multiply            (MaskedRasterHandle const& raster,
                                        double value);

} // namespace python
} // namespace fern
