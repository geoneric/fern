#pragma once
#include <boost/python.hpp>
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

boost::python::object
                   slope               (fern::python::MaskedRasterHandle
                                            const& dem);

} // namespace python
} // namespace fern
