// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
