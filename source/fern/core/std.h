// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <algorithm>


namespace fern {

template<
    class Container>
void sort(
    Container& container)
{
    std::sort(container.begin(), container.end());
}


template<
    class Container,
    class Predicate>
void sort(
    Container& container,
    Predicate predicate)
{
    std::sort(container.begin(), container.end(), predicate);
}

} // namespace fern
