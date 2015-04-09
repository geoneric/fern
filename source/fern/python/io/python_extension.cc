// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include <boost/python.hpp>
#include <gdal_priv.h>
#include "fern/python/core/init_python_module.h"
#include "fern/python/io/gdal.h"


namespace bp = boost::python;
namespace fp = fern::python;

BOOST_PYTHON_MODULE(_fern_io)
{
    INIT_PYTHON_MODULE("_fern_io")

    // Don't throw in case of an error.
    CPLSetErrorHandler(CPLQuietErrorHandler);

    GDALAllRegister();

    bp::def("read_raster", fp::read_raster,
        "Read raster given the name passed in and return a reference to\n"
        "the result.",
        bp::arg("name"))
        ;
    bp::def("write_raster", fp::write_raster,
        "Write raster given the raster, name and format passed in.",
        (bp::arg("raster"), bp::arg("name"), bp::arg("format")))
        ;
}
