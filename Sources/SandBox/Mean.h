#ifndef INCLUDED_BOOST_STATIC_ASSERT
#include <boost/static_assert.hpp>
#define INCLUDED_BOOST_STATIC_ASSERT
#endif

#ifndef INCLUDED_BOOST_TYPE_TRAITS_IS_ARITHMETIC
#include <boost/type_traits/is_arithmetic.hpp>
#define INCLUDED_BOOST_TYPE_TRAITS_IS_ARITHMETIC
#endif

#ifndef INCLUDED_BOOST_TYPE_TRAITS_IS_FLOATING_POINT
#include <boost/type_traits/is_floating_point.hpp>
#define INCLUDED_BOOST_TYPE_TRAITS_IS_FLOATING_POINT
#endif



namespace algorithms {

template<typename Argument, typename Result=Argument>
class Mean
{

private:

  BOOST_STATIC_ASSERT(boost::is_arithmetic<Argument>::value);
  BOOST_STATIC_ASSERT(boost::is_floating_point<Result>::value);

  Result           _sum;

  size_t           _count;

public:

  void init(
         Argument const& value)
  {
    _sum = value;
    _count = 1;
  }

  inline void operator+=(
         Argument const& value)
  {
    assert(_count > 0);
    _sum += value;
    ++_count;
  }

  void operator()(
         Result& result)
  {
    assert(_count > 0);
    result = _sum / _count;
  }
};

} // namespace algorithms
