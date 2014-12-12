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
    fern::MaskedRaster<T2, 2> const& rhs,
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
void iadd(
    fern::MaskedRaster<T1, 2>& self,
    fern::MaskedRaster<T2, 2> const& other)
{
    // TODO Assert raster properties.
    merge_no_data(other, self);
    add(self, other, self);
}

} // Anonymous namespace


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

} // namespace python
} // namespace fern
