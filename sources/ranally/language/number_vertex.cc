#include "ranally/language/number_vertex.h"


namespace ranally {

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

} // namespace ranally
