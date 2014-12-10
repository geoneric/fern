#pragma once
#include <boost/python.hpp>


namespace fern {
namespace python {

boost::python::object
                   read_raster         (boost::python::str pathname_object);

} // namespace python
} // namespace fern
