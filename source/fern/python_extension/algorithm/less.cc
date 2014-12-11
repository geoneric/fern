#include "fern/python_extension/algorithm/less.h"
#include "fern/python_extension/algorithm/merge_no_data.h"
#include "fern/python_extension/algorithm/unite_no_data.h"
#include "fern/core/constant_traits.h"
#include "fern/algorithm/algebra/elementary/less.h"
#include "fern/python_extension/core/switch_on_value_type.h"


namespace fa = fern::algorithm;


namespace fern {
namespace python {
namespace {

template<
    typename T1,
    typename T2,
    typename R>
void less(
    fern::MaskedRaster<T1, 2> const& lhs,
    fern::MaskedRaster<T2, 2> const& rhs,
    fern::MaskedRaster<R, 2>& result)
{
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    InputNoDataPolicy input_no_data_policy(result.mask(), true);
    OutputNoDataPolicy output_no_data_policy(result.mask(), true);

    fa::algebra::less(input_no_data_policy, output_no_data_policy,
        algorithm::parallel, lhs, rhs, result);
}


template<
    typename T1,
    typename T2,
    typename R>
void less(
    fern::MaskedRaster<T1, 2> const& lhs,
    T2 const& rhs,
    fern::MaskedRaster<R, 2>& result)
{
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    InputNoDataPolicy input_no_data_policy(result.mask(), true);
    OutputNoDataPolicy output_no_data_policy(result.mask(), true);

    fa::algebra::less(input_no_data_policy, output_no_data_policy,
        algorithm::parallel, lhs, rhs, result);
}


template<
    typename T1,
    typename T2,
    typename R>
void less(
    T1 const& lhs,
    fern::MaskedRaster<T2, 2> const& rhs,
    fern::MaskedRaster<R, 2>& result)
{
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    InputNoDataPolicy input_no_data_policy(result.mask(), true);
    OutputNoDataPolicy output_no_data_policy(result.mask(), true);

    fa::algebra::less(input_no_data_policy, output_no_data_policy,
        algorithm::parallel, lhs, rhs, result);
}


template<
    typename T1,
    typename T2>
MaskedRasterHandle less(
    fern::MaskedRaster<T1, 2> const& lhs,
    fern::MaskedRaster<T2, 2> const& rhs)
{
    // TODO Assert raster properties.
    auto sizes = extents[lhs.shape()[0]][lhs.shape()[1]];
    using R = uint8_t;
    auto handle = std::make_shared<fern::MaskedRaster<R, 2>>(sizes,
        lhs.transformation());
    unite_no_data(lhs, rhs, *handle);
    less(lhs, rhs, *handle);
    return std::make_shared<MaskedRaster>(handle);
}


template<
    typename T1,
    typename T2>
MaskedRasterHandle less(
    fern::MaskedRaster<T1, 2> const& lhs,
    T2 const& rhs)
{
    auto sizes = extents[lhs.shape()[0]][lhs.shape()[1]];
    using R = uint8_t;
    auto handle = std::make_shared<fern::MaskedRaster<R, 2>>(sizes,
        lhs.transformation());
    merge_no_data(lhs, *handle);
    less(lhs, rhs, *handle);
    return std::make_shared<MaskedRaster>(handle);
}


template<
    typename T1,
    typename T2>
MaskedRasterHandle less(
    T1 const& lhs,
    fern::MaskedRaster<T2, 2> const& rhs)
{
    auto sizes = extents[rhs.shape()[0]][rhs.shape()[1]];
    using R = uint8_t;
    auto handle = std::make_shared<fern::MaskedRaster<R, 2>>(sizes,
        rhs.transformation());
    merge_no_data(rhs, *handle);
    less(lhs, rhs, *handle);
    return std::make_shared<MaskedRaster>(handle);
}

} // Anonymous namespace


#define CASE2(                            \
    value_type_enum2,                     \
    value_type2,                          \
    value_type1)                          \
case value_type_enum2: {                  \
    result = less(                    \
        raster1->raster<value_type1>(),   \
        raster2->raster<value_type2>());  \
    break;                                \
}

#define CASE1(                                                  \
    value_type_enum1,                                           \
    value_type1,                                                \
    value_type_enum2)                                           \
case value_type_enum1: {                                        \
    SWITCH_ON_VALUE_TYPE2(value_type_enum2, CASE2, value_type1) \
    break;                                                      \
}

MaskedRasterHandle less(
    MaskedRasterHandle const& raster1,
    MaskedRasterHandle const& raster2)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE1(raster1->value_type(), CASE1, raster2->value_type());
    return result;
}

#undef CASE1
#undef CASE2


#define CASE(                            \
    value_type_enum2,                    \
    value_type2)                         \
case value_type_enum2: {                 \
    result = less(                       \
        value,                           \
        raster->raster<value_type2>());  \
    break;                               \
}

MaskedRasterHandle less(
    int64_t value,
    MaskedRasterHandle const& raster)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE(raster->value_type(), CASE)
    return result;
}


MaskedRasterHandle less(
    double value,
    MaskedRasterHandle const& raster)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE(raster->value_type(), CASE)
    return result;
}

#undef CASE


#define CASE(                           \
    value_type_enum1,                   \
    value_type1)                        \
case value_type_enum1: {                \
    result = less(                      \
        raster->raster<value_type1>(),  \
        value);                         \
    break;                              \
}

MaskedRasterHandle less(
    MaskedRasterHandle const& raster,
    int64_t value)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE(raster->value_type(), CASE)
    return result;
}


MaskedRasterHandle less(
    MaskedRasterHandle const& raster,
    double value)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE(raster->value_type(), CASE)
    return result;
}

#undef CASE

} // namespace python
} // namespace fern
