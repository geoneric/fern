#ifndef INCLUDED_ALGORITHM
#include <algorithm>
#define INCLUDED_ALGORITHM
#endif

#ifndef INCLUDED_BOOST_TYPE_TRAITS_IS_ARITHMETIC
#include <boost/type_traits/is_arithmetic.hpp>
#define INCLUDED_BOOST_TYPE_TRAITS_IS_ARITHMETIC
#endif

#ifndef INCLUDED_BOOST_TYPE_TRAITS_IS_SAME
#include <boost/type_traits/is_same.hpp>
#define INCLUDED_BOOST_TYPE_TRAITS_IS_SAME
#endif



namespace algorithms {

template<typename Argument, typename Result=Argument>
class Max
{
private:

  BOOST_STATIC_ASSERT(boost::is_arithmetic<Argument>::value);
  BOOST_STATIC_ASSERT((boost::is_same<Argument, Result>::value));

  Result           _max;

public:

  void init(
         Argument const& value)
  {
    _max = value;
  }

  inline void operator+=(
         Argument const& value)
  {
    _max = std::max<Argument>(_max, value);
  }

  void operator()(
         Result& result)
  {
    result = _max;
  }
};

} // namespace algorithms
