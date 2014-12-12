#include "fern/python_extension/algorithm/algebra/elementary/add.h"
#include "fern/python_extension/algorithm/core/merge_no_data.h"
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/python_extension/core/switch_on_value_type.h"

// #include "fern/feature/core/array_traits.h"
// #include "fern/feature/core/masked_raster_traits.h"


namespace fa = fern::algorithm;


namespace fern {
namespace python {
namespace {

template<
    typename T1,
    typename T2,
    typename R>
void add(
    fern::MaskedRaster<T1, 2> const& lhs,
    T2 const& rhs,
    fern::MaskedRaster<R, 2>& result)
{
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    InputNoDataPolicy input_no_data_policy(result.mask(), true);
    OutputNoDataPolicy output_no_data_policy(result.mask(), true);

    fa::algebra::add<fa::add::OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, algorithm::sequential, lhs, rhs, result);
}


template<
    typename T1,
    typename T2>
MaskedRasterHandle add(
    fern::MaskedRaster<T1, 2> const& lhs,
    T2 const& rhs)
{
    auto sizes = extents[lhs.shape()[0]][lhs.shape()[1]];
    using R = algorithm::add::result_type<T1, T2>;
    auto handle = std::make_shared<fern::MaskedRaster<R, 2>>(sizes,
        lhs.transformation());
    merge_no_data(lhs, *handle);
    add(lhs, rhs, *handle);
    return std::make_shared<MaskedRaster>(handle);
}

} // Anonymous namespace

#define CASE(                           \
    value_type_enum1,                   \
    value_type1)                        \
case value_type_enum1: {                \
    result = add(                       \
        raster->raster<value_type1>(),  \
        value);                         \
    break;                              \
}

MaskedRasterHandle add(
    MaskedRasterHandle const& raster,
    int64_t value)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE(raster->value_type(), CASE)
    return result;
}


MaskedRasterHandle add(
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
