#ifndef INCLUDED_ITERATOR
#include <iterator>
#define INCLUDED_ITERATOR
#endif

#ifndef INCLUDED_BOOST_FOREACH
#include <boost/foreach.hpp>
#define INCLUDED_BOOST_FOREACH
#endif

#ifndef INCLUDED_BOOST_FUNCTION
#include <boost/function.hpp>
#define INCLUDED_BOOST_FUNCTION
#endif



struct DontMask
{
  static inline bool mask(
         size_t /* i */)
  {
    // Don't mask anything.
    return false;
  }
};



//!
/*!
  \tparam    InputIterator Iterator type of collection of values.
  \warning   .
  \sa        .
*/
template <class InputIterator>
class MaskByValuePredicate
{
private:

  //! Type of function to use for checking values. Unary predicate.
  typedef boost::function<bool (
         typename std::iterator_traits<InputIterator>::value_type)> Predicate;

  //! Copy of predicate.
  Predicate        _predicate;

  //! Collection of iterators to containers with values. For use only.
  std::vector<InputIterator> _iterators;

public:

  void setMaskPredicate(
         Predicate const& predicate)
  {
    _predicate = predicate;
  }

  //! Sets the collection of values to test the predicate against.
  /*!
    \param     values Iterator to a collection of values.
    \warning   Previous contents in the layered collection of input iterators
               are lost.

    For use in unary operations.
  */
  void setValues(
         InputIterator values)
  {
    _iterators.clear();
    _iterators.push_back(values);
  }

  //! Sets the a collections of values to test the predicate against.
  /*!
    \param     values1 Iterator to a collection of values.
    \param     values2 Iterator to a collection of values.
    \warning   Previous contents in the layered collection of input iterators
               are lost.

    For use in binary operations.
  */
  void setValues(
         InputIterator values1,
         InputIterator values2)
  {
    _iterators.clear();
    _iterators.push_back(values1);
    _iterators.push_back(values2);
  }

  //! Sets the a collections of values to test the predicate against.
  /*!
    \param     collections Iterator to a collection of iterators to values.
    \param     nrCollections Number of collections that should be set.
    \warning   Previous contents in the layered collection of input iterators
               are lost.

    For use in n-ary operations.
  */
  template<class InputIteratorCollectionIterator>
  void setValues(
         InputIteratorCollectionIterator collections,
         size_t nrCollections)
  {
    _iterators.clear();

    for(size_t i = 0; i < nrCollections; ++i) {
      _iterators.push_back(*collections);
      ++collections;
    }
  }

  inline bool mask(
         size_t i)
  {
    bool result = false;

    BOOST_FOREACH(InputIterator it, _iterators) {
      if(_predicate(*(it + i))) {
        result = true;
        break;
      }
    }

    return result;
  }
};

