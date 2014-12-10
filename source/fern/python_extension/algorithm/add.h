#pragma once
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle&
                   iadd                (MaskedRasterHandle& self,
                                        MaskedRasterHandle const& other);

MaskedRasterHandle add                 (MaskedRasterHandle const& raster1,
                                        MaskedRasterHandle const& raster2);

} // namespace python
} // namespace fern
