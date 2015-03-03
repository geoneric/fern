#pragma once
#include <memory>
#include <utility>
#include <boost/python.hpp>
#include <boost/any.hpp>
#include "fern/core/type_traits.h"
#include "fern/python/feature/detail/masked_raster.h"


namespace fern {
namespace python {

class MaskedRaster
{

public:

    template<
        typename T>
    explicit       MaskedRaster        (std::shared_ptr<
                                            detail::MaskedRaster<T>> pointer);

                   MaskedRaster        (boost::python::tuple const& sizes,
                                        boost::python::tuple const& origin,
                                        boost::python::tuple const& cell_sizes,
                                        ValueType value_type);

                   MaskedRaster        (boost::python::list& values,
                                        boost::python::list& mask,
                                        boost::python::tuple& origin,
                                        boost::python::tuple& cell_sizes,
                                        ValueType value_type);

    boost::python::tuple
                   sizes               () const;

    boost::python::tuple
                   origin              () const;

    boost::python::tuple
                   cell_sizes          () const;

    ValueType      value_type          () const;

    template<
        typename T>
    detail::MaskedRaster<T> &
                   raster              ();

    template<
        typename T>
    detail::MaskedRaster<T> const&
                   raster              () const;

private:

    std::tuple<size_t, size_t> _sizes;

    std::tuple<double, double> _origin;

    std::tuple<double, double> _cell_sizes;

    ValueType _value_type;

    boost::any _pointer;

};


using MaskedRasterHandle = std::shared_ptr<MaskedRaster>;


template<
    typename T>
inline MaskedRaster::MaskedRaster(
    std::shared_ptr<detail::MaskedRaster<T>> pointer)

    : _sizes(pointer->sizes()),
      _origin(pointer->origin()),
      _cell_sizes(pointer->cell_sizes()),
      _value_type(TypeTraits<T>::value_type),
      _pointer(pointer)

{
}


template<
    typename T>
inline detail::MaskedRaster<T>& MaskedRaster::raster()
{
    using Pointer = std::shared_ptr<detail::MaskedRaster<T>>;
    return *boost::any_cast<Pointer>(_pointer);
}


template<
    typename T>
inline detail::MaskedRaster<T> const& MaskedRaster::raster() const
{
    using Pointer = std::shared_ptr<detail::MaskedRaster<T>>;
    return *boost::any_cast<Pointer>(_pointer);
}

} // namespace python
} // namespace fern
