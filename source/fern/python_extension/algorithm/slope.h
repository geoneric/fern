#pragma once
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

boost::python::object
                   slope               (MaskedRasterHandle const& dem);

} // namespace python
} // namespace fern
