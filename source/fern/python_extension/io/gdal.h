#pragma once
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle read_raster         (std::string const& name);

void               write_raster        (MaskedRasterHandle const& raster,
                                        std::string const& name,
                                        std::string const& format);

} // namespace python
} // namespace fern
