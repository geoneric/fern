// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/distance.hpp>
#include "fern/language/operation/data_traits.h"


namespace fern {
namespace language {
namespace detail {

template<
    typename Argument>
inline void print(
    ConstantTag /* tag */,
    Argument argument,
    std::ostream& stream)
{
    stream << argument << '\n';
}


template<
    typename Argument>
inline void print(
    RangeTag /* tag */,
    Argument const& argument,
    std::ostream& stream)
{
    stream << '[';

    size_t distance = boost::distance(argument);

    if(distance > 0) {
        typename boost::range_iterator<Argument const>::type pos =
            boost::const_begin(argument);
        typename boost::range_iterator<Argument const>::type end =
            boost::const_end(argument);

        if(distance < 7) {
            stream << *pos++;
            while(pos != end) {
                stream << ", " << *pos++;
            }
        }
        else {
            stream << *pos++ << ", ";
            stream << *pos++ << ", ";
            stream << *pos++ << ", ..., ";
            stream << *(end - 3) << ", " << *(end - 2) << ", " << *(end - 1);
        }
    }

    stream << "]\n";
}


template<
    typename Argument>
inline void print(
    RasterTag /* tag */,
    Argument const& /* argument */,
    std::ostream& stream)
{
    stream << '[';

    stream << "]\n";
}

} // namespace detail


template<
    typename Argument>
inline void print(
    Argument const& argument,
    std::ostream& stream)
{
    using category = typename DataTraits<Argument>::DataCategory;
    detail::print(category(), argument, stream);
}

} // namespace language
} // namespace fern
