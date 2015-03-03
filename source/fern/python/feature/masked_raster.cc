#include "fern/python/feature/masked_raster.h"
#include "fern/python/core/switch_on_value_type.h"


namespace fern {
namespace python {

template<
    typename T>
static void copy(
    boost::python::list& values,
    boost::python::list& mask,
    detail::MaskedRaster<T>& masked_raster)
{
    namespace bp = boost::python;

    size_t const size1{std::get<0>(masked_raster.sizes())};
    size_t const size2{std::get<1>(masked_raster.sizes())};
    size_t index_;

    for(size_t index1 = 0; index1 < size1; ++index1) {
        bp::list row_object(values[index1]);
        bp::list mask_row_object(mask[index1]);

        index_ = masked_raster.index(index1, 0);

        for(size_t index2 = 0; index2 < size2; ++index2) {
            bp::object mask_value_object(mask_row_object[index2]);

            if(bp::extract<bool>(mask_value_object)) {
                set_no_data(masked_raster.element(index_));
            }
            else {
                bp::object value_object(row_object[index2]);
                masked_raster.element(index_) = bp::extract<T>(value_object);
            }

            ++index_;
        }
    }
}


#define CASE(                                                       \
    value_type_enum,                                                \
    value_type)                                                     \
case value_type_enum: {                                             \
    _pointer = std::make_shared<detail::MaskedRaster<value_type>>(  \
        _sizes, _origin, _cell_sizes, value_type{0});               \
    break;                                                          \
}

MaskedRaster::MaskedRaster(
    boost::python::tuple const& sizes,
    boost::python::tuple const& origin,
    boost::python::tuple const& cell_sizes,
    ValueType value_type)

    : _sizes(boost::python::extract<size_t>(sizes[0])(),
          boost::python::extract<size_t>(sizes[1])()),
      _origin(boost::python::extract<double>(origin[0])(),
          boost::python::extract<double>(origin[1])()),
      _cell_sizes(boost::python::extract<double>(cell_sizes[0])(),
          boost::python::extract<double>(cell_sizes[1])()),
      _value_type(value_type)

{
    SWITCH_ON_VALUE_TYPE(value_type, CASE);

    assert(!_pointer.empty());
}

#undef CASE


#define CASE(                                                                 \
    value_type_enum, value_type)                                              \
case value_type_enum: {                                                       \
    auto masked_raster = std::make_shared<detail::MaskedRaster<value_type>>(  \
        _sizes, _origin, _cell_sizes);                                        \
    _pointer = masked_raster;                                                 \
    copy(values, mask, *masked_raster);                                       \
    break;                                                                    \
}

MaskedRaster::MaskedRaster(
    boost::python::list& values,
    boost::python::list& mask,
    boost::python::tuple& origin,
    boost::python::tuple& cell_sizes,
    ValueType value_type)

    : _sizes(
          boost::python::extract<size_t>(values.attr("__len__")()),
          boost::python::extract<size_t>(values[0].attr("__len__")())),
      _origin(boost::python::extract<double>(origin[0])(),
          boost::python::extract<double>(origin[1])()),
      _cell_sizes(boost::python::extract<double>(cell_sizes[0])(),
          boost::python::extract<double>(cell_sizes[1])()),
      _value_type(value_type)

{
    namespace bp = boost::python;

    SWITCH_ON_VALUE_TYPE(value_type, CASE);

    assert(!_pointer.empty());
}

#undef CASE


boost::python::tuple MaskedRaster::sizes() const
{
    return boost::python::make_tuple(std::get<0>(_sizes), std::get<1>(_sizes));
}


boost::python::tuple MaskedRaster::origin() const
{
    return boost::python::make_tuple(std::get<0>(_origin),
        std::get<1>(_origin));
}


boost::python::tuple MaskedRaster::cell_sizes() const
{
    return boost::python::make_tuple(std::get<0>(_cell_sizes),
        std::get<1>(_cell_sizes));
}


ValueType MaskedRaster::value_type() const
{
    return _value_type;
}

} // namespace python
} // namespace fern
