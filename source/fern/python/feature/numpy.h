#pragma once
#include <boost/python.hpp>


namespace fern {
namespace python {

class MaskedRaster;

boost::python::object
                   raster_as_numpy_array
                                       (fern::python::MaskedRaster const&
                                            masked_raster);

// boost::python::object
//                    mask_as_numpy_array (fern::python::MaskedRaster const&
//                                             masked_raster);

} // namespace python
} // namespace fern
