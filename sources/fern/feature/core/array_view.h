#pragma once
#include <boost/multi_array.hpp>


namespace fern {
namespace detail {

template<
    class T,
    size_t nr_dimensions>
using boost_array_view = typename boost::array_view_gen<
    boost::multi_array<T, nr_dimensions>, nr_dimensions>::type;

} // namespace detail


template<
    class T,
    size_t nr_dimensions>
class ArrayView:
    public detail::boost_array_view<T, nr_dimensions>
{

public:

                   ArrayView           (detail::boost_array_view<T,
                                            nr_dimensions> const& view);

};


template<
    class T,
    size_t nr_dimensions>
inline ArrayView<T, nr_dimensions>::ArrayView(
    detail::boost_array_view<T, nr_dimensions> const& view)

    : detail::boost_array_view<T, nr_dimensions>(view)

{
}

} // namespace fern
