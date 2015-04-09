// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <boost/multi_array.hpp>


namespace fern {
namespace detail {

template<
    typename T,
    size_t nr_dimensions>
using boost_array_view = typename boost::array_view_gen<
    boost::multi_array<T, nr_dimensions>, nr_dimensions>::type;


// template<
//     typename T,
//     size_t nr_dimensions>
// using boost_const_array_view = typename boost::const_array_view_gen<
//     boost::multi_array<T, nr_dimensions>, nr_dimensions>::type;

} // namespace detail


template<
    typename T,
    size_t nr_dimensions>
class ArrayView:
    public detail::boost_array_view<T, nr_dimensions>
{

public:

                   ArrayView           (detail::boost_array_view<T,
                                            nr_dimensions> const& view);

};


template<
    typename T,
    size_t nr_dimensions>
inline ArrayView<T, nr_dimensions>::ArrayView(
    detail::boost_array_view<T, nr_dimensions> const& view)

    : detail::boost_array_view<T, nr_dimensions>(view)

{
}

} // namespace fern
