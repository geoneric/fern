#pragma once
#include <utility>
#include <boost/python.hpp>
#include <boost/any.hpp>
#include "fern/core/value_type.h"


namespace fern {
namespace python {

class MaskedRaster
{

public:

                   MaskedRaster        (boost::python::tuple const& sizes,
                                        boost::python::tuple const& origin,
                                        boost::python::tuple const& cell_sizes,
                                        fern::ValueType value_type);

                   MaskedRaster        (boost::python::list& values,
                                        boost::python::list& mask,
                                        boost::python::tuple& origin,
                                        boost::python::tuple& cell_sizes,
                                        fern::ValueType value_type);

    boost::python::tuple
                   sizes               () const;

    boost::python::tuple
                   origin              () const;

    boost::python::tuple
                   cell_sizes          () const;

private:

    std::pair<size_t, size_t> const _sizes;

    std::pair<double, double> const _origin;

    std::pair<double, double> const _cell_sizes;

    fern::ValueType const _value_type;

    boost::any _masked_raster;

};

} // namespace python
} // namespace fern
