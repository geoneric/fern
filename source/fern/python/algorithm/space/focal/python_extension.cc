// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include <boost/python.hpp>
#include "fern/python/core/init_python_module.h"
#include "fern/python/algorithm/space/focal/slope.h"


namespace bp = boost::python;
namespace fp = fern::python;


BOOST_PYTHON_MODULE(_fern_algorithm_space_focal)
{
    INIT_PYTHON_MODULE("_fern_algorithm_space_focal")

    bp::def("slope", fp::slope);
}
