#pragma once
#include <memory>
#include <utility>
#include <boost/python.hpp>
#include <boost/any.hpp>
#include "fern/core/type_traits.h"
#include "fern/feature/core/masked_raster.h"


namespace fern {
namespace python {

class MaskedRaster
{

public:

    // template<
    //     typename T>
    // explicit       MaskedRaster        (fern::MaskedRaster<T, 2>::
    //                                         Transformation const&
    //                                             transformation);

    ///                MaskedRaster        ();

    ///                MaskedRaster        (MaskedRaster& raster);

    /// MaskedRaster&  operator=           (MaskedRaster&& raster);

    template<
        typename T>
    explicit       MaskedRaster        (std::shared_ptr<
                                            fern::MaskedRaster<T, 2>> pointer);

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
    fern::MaskedRaster<T, 2> const&
                   raster              () const;

private:

    std::pair<size_t, size_t> _sizes;

    std::pair<double, double> _origin;

    std::pair<double, double> _cell_sizes;

    ValueType _value_type;

    boost::any _pointer;

};


using MaskedRasterHandle = std::shared_ptr<MaskedRaster>;


// template<
//     typename T>
// inline MaskedRaster::MaskedRaster(
//     fern::MaskedRaster<T, 2>::Transformation const& transformation)
// 
//     : _sizes(pointer->shape()[0], pointer->shape()[1]),
//       _origin(transformation()[0], transformation()[2]),
//       _cell_sizes(transformation()[1], transformation()[3]),
//       _value_type(TypeTraits<T>::value_type),
//       _pointer(std::make_shared<fern::MaskedRaster<T, 2>>)
// 
// {
// }


template<
    typename T>
inline MaskedRaster::MaskedRaster(
    std::shared_ptr<fern::MaskedRaster<T, 2>> pointer)

    : _sizes(pointer->shape()[0], pointer->shape()[1]),
      _origin(pointer->transformation()[0], pointer->transformation()[2]),
      _cell_sizes(pointer->transformation()[1], pointer->transformation()[3]),
      _value_type(TypeTraits<T>::value_type),
      _pointer(pointer)

{
}


template<
    typename T>
inline fern::MaskedRaster<T, 2> const& MaskedRaster::raster() const
{
    using Pointer = std::shared_ptr<fern::MaskedRaster<T, 2>>;
    return *boost::any_cast<Pointer>(_pointer);
}

} // namespace python
} // namespace fern
