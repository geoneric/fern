// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/number_vertex.h"


namespace fern {

template<typename T>
NumberVertex<T>::NumberVertex(
    T value)

    : ExpressionVertex(String(boost::format("%1%") % value)),
      _value(value)

{
}


template<typename T>
NumberVertex<T>::NumberVertex(
    int line_nr,
    int col_id,
    T value)

    : ExpressionVertex(line_nr, col_id, String(boost::format("%1%") % value)),
      _value(value)

{
}


template<typename T>
T NumberVertex<T>::value() const
{
    return _value;
}


template class NumberVertex<int8_t>;
template class NumberVertex<int16_t>;
template class NumberVertex<int32_t>;
template class NumberVertex<int64_t>;
template class NumberVertex<uint8_t>;
template class NumberVertex<uint16_t>;
template class NumberVertex<uint32_t>;
template class NumberVertex<uint64_t>;
template class NumberVertex<float>;
template class NumberVertex<double>;

} // namespace fern
