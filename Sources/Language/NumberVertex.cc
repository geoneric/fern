#include "NumberVertex.h"



namespace ranally {

template<typename T>
NumberVertex<T>::NumberVertex(
  T value)

  : ExpressionVertex(),
    _value(value)

{
}



template<typename T>
NumberVertex<T>::NumberVertex(
  int lineNr,
  int colId,
  T value)

  : ExpressionVertex(lineNr, colId),
    _value(value)

{
}



template<typename T>
NumberVertex<T>::~NumberVertex()
{
}



template class NumberVertex<int>;
template class NumberVertex<long long>;
template class NumberVertex<double>;

} // namespace ranally

