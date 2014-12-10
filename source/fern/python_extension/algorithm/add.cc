#include "fern/python_extension/algorithm/add.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_raster_traits.h"
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/algorithm/core/unite_no_data.h"
#include "fern/python_extension/core/switch_on_value_type.h"


namespace fa = fern::algorithm;


namespace fern {
namespace python {


template<
    typename T1,
    typename T2,
    typename R>
void unite_no_data(
    fern::MaskedRaster<T1, 2> const& raster1,
    fern::MaskedRaster<T2, 2> const& raster2,
    fern::MaskedRaster<R, 2>& result_raster)
{
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    fa::SkipNoData<
        InputNoDataPolicy,
        InputNoDataPolicy> input_no_data_policy(
            InputNoDataPolicy(raster1.mask(), true),
            InputNoDataPolicy(raster2.mask(), true));
    OutputNoDataPolicy output_no_data_policy(result_raster.mask(), true);

    fa::core::unite_no_data(input_no_data_policy,
        output_no_data_policy, algorithm::parallel, raster1, raster2,
        result_raster);
}


template<
    typename T1,
    typename T2,
    typename R>
void add(
    fern::MaskedRaster<T1, 2> const& lhs,
    fern::MaskedRaster<T2, 2> const& rhs,
    fern::MaskedRaster<R, 2>& result)
{
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    InputNoDataPolicy input_no_data_policy(result.mask(), true);
    OutputNoDataPolicy output_no_data_policy(result.mask(), true);

    fa::algebra::add<fa::add::OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, algorithm::parallel, lhs, rhs, result);
}


template<
    typename T1,
    typename T2>
fern::MaskedRaster<T1, 2>& iadd(
    fern::MaskedRaster<T1, 2>& self,
    fern::MaskedRaster<T2, 2> const& other)
{
    // TODO Assert raster properties.
    merge_no_data(self, other, self);
    add(self, other, self);
    return self;
}


template<
    typename T1,
    typename T2>
MaskedRasterHandle add(
    fern::MaskedRaster<T1, 2> const& lhs,
    fern::MaskedRaster<T2, 2> const& rhs)
{
    // TODO Assert raster properties.
    auto sizes = extents[lhs.shape()[0]][lhs.shape()[1]];
    using R = algorithm::add::result_type<T1, T2>;
    auto handle = std::make_shared<fern::MaskedRaster<R, 2>>(sizes,
        lhs.transformation());
    unite_no_data(lhs, rhs, *handle);
    add(lhs, rhs, *handle);
    return std::make_shared<MaskedRaster>(handle);
}


#define CASE2(                          \
    value_type_enum2,                   \
    value_type2,                        \
    value_type1)                        \
case value_type_enum2: {                \
    iadd(self->raster<value_type1>(),   \
        other->raster<value_type2>());  \
    break;                              \
}

#define CASE1(                                                   \
    value_type_enum1,                                            \
    value_type1,                                                 \
    value_type_enum2)                                            \
case value_type_enum1: {                                         \
    SWITCH_ON_VALUE_TYPE2(value_type_enum2, CASE2, value_type1)  \
    break;                                                       \
}

MaskedRasterHandle& iadd(
    MaskedRasterHandle& self,
    MaskedRasterHandle const& other)
{
    SWITCH_ON_VALUE_TYPE1(self->value_type(), CASE1, other->value_type());
    return self;
}

#undef CASE1
#undef CASE2


#define CASE2(                                    \
    value_type_enum2,                             \
    value_type2,                                  \
    value_type1)                                  \
case value_type_enum2: {                          \
    result = add(raster1->raster<value_type1>(),  \
        raster2->raster<value_type2>());          \
    break;                                        \
}

#define CASE1(                                                  \
    value_type_enum1,                                           \
    value_type1,                                                \
    value_type_enum2)                                           \
case value_type_enum1: {                                        \
    SWITCH_ON_VALUE_TYPE2(value_type_enum2, CASE2, value_type1) \
    break;                                                      \
}

MaskedRasterHandle add(
    MaskedRasterHandle const& raster1,
    MaskedRasterHandle const& raster2)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE1(raster1->value_type(), CASE1, raster2->value_type());
    return result;
}

#undef CASE1
#undef CASE2

} // namespace python
} // namespace fern
