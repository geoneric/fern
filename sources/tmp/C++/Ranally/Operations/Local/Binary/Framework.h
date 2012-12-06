#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_FRAMEWORK
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_FRAMEWORK

#ifndef INCLUDED_ITERATOR
#include <iterator>
#define INCLUDED_ITERATOR
#endif

#ifndef INCLUDED_BOOST_MATH_SPECIAL_FUNCTIONS_FPCLASSIFY
#include <boost/math/special_functions/fpclassify.hpp>
#define INCLUDED_BOOST_MATH_SPECIAL_FUNCTIONS_FPCLASSIFY
#endif



namespace ranally {
namespace operations {
namespace binary {
namespace framework {


// TODO Let BinarySame use BinaryDifferent as a base class.
// TODO Add ClipMaskPolicy.


//! Template class for binary local operations with equal argument and result types.
/*!
  \tparam    T Type of argument and result values.
  \tparam    NoData Type of no-data values.
  \tparam    Algorithm Template class for creating algorithm class.
  \tparam    DomainPolicy Template class for creating domain policy class.
  \tparam    RangePolicy Template class for creating range policy class.
  \tparam    ClipMaskPolicy Template class for creating clip mask policy class.
  \tparam    NoDataPolicy Template class for creating no-data policy class.

  The policies are all default constructed and can be configured by calling
  their respective access functions.
*/
template<typename T,
         typename NoData,
         typename ClipMask,
         template<typename> class Algorithm,
         template<typename> class DomainPolicy,
         template<typename> class RangePolicy,
         template<typename> class ClipMaskPolicy,
         template<typename> class NoDataPolicy
>
class BinarySame
{
private:

  Algorithm<T>     _algorithm;

  DomainPolicy<T>  _domainPolicy;

  RangePolicy<T>   _rangePolicy;

  ClipMaskPolicy<ClipMask> _clipMaskPolicy;

  NoDataPolicy<NoData> _noDataPolicy;

public:

  DomainPolicy<T>& domainPolicy() const
  {
    return _domainPolicy;
  }

  RangePolicy<T>& rangePolicy() const
  {
    return _rangePolicy;
  }

  ClipMaskPolicy<ClipMask>& clipMaskPolicy() const
  {
    return _clipMaskPolicy;
  }

  NoDataPolicy<NoData>& noDataPolicy() const
  {
    return _noDataPolicy;
  }

  //!
  /*!
    \tparam    .
    \param     .
    \return    .
    \exception .
    \warning   .
    \sa        .

    operation(scalar, scalar, scalar)
  */
  inline void operator()(
         T argument1,
         T argument2,
         T& result,
         ClipMask clipMask,
         NoData& noData) const
  {
    if(!_clipMaskPolicy.mask(clipMask)) {
      if(!_noDataPolicy.isNoData(noData)) {
        if(!_domainPolicy.inDomain(argument1, argument2)) {
          _noDataPolicy.setNoData(noData);
        }
        else {
          assert(!boost::math::isnan(argument1));
          assert(!boost::math::isnan(argument2));

          result = _algorithm.calculate(argument1, argument2);

          if(!_rangePolicy.inRange(argument1, argument2, result)) {
            _noDataPolicy.setNoData(noData);
          }
        }
      }
    }
  }

  //!
  /*!
    \tparam    .
    \param     .
    \return    .
    \exception .
    \warning   .
    \sa        .

    operation(scalar, scalar*, scalar*)
  */
  template<
         class ArgumentIterator,
         class ResultIterator,
         class ClipMaskIterator,
         class NoDataIterator>
  inline void operator()(
         T argument1,
         ArgumentIterator argument2,
         ResultIterator result,
         ClipMaskIterator clipMask,
         NoDataIterator noData,
         size_t nrValues) const
  {
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<ArgumentIterator>::value_type,
         T>::value));
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<ResultIterator>::value_type,
         T>::value));
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<ClipMaskIterator>::value_type,
         ClipMask>::value));
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<NoDataIterator>::value_type,
         NoData>::value));

    for(size_t i = 0; i < nrValues; ++i, ++argument2, ++result,
         ++clipMask, ++noData) {
      if(!clipMaskPolicy.mask(*clipMask)) {
        if(!_noDataPolicy.isNoData(*noData)) {
          if(!_domainPolicy.inDomain(argument1, *argument2)) {
            _noDataPolicy.setNoData(*noData);
          }
          else {
            assert(!boost::math::isnan(argument1));
            assert(!boost::math::isnan(*argument2));

            *result = _algorithm(argument1, *argument2);

            if(!_rangePolicy.inRange(argument1, *argument2, *result)) {
              _noDataPolicy.setNoData(*noData);
            }
          }
        }
      }
    }
  }

  //!
  /*!
    \tparam    .
    \param     .
    \return    .
    \exception .
    \warning   .
    \sa        .

    operation(scalar*, scalar, scalar*)
  */
  template<
         class ArgumentIterator,
         class ResultIterator,
         class ClipMaskIterator,
         class NoDataIterator>
  inline void operator()(
         ArgumentIterator argument1,
         T argument2,
         ResultIterator result,
         ClipMaskIterator clipMask,
         NoDataIterator noData,
         size_t nrValues) const
  {
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<ArgumentIterator>::value_type,
         T>::value));
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<ResultIterator>::value_type,
         T>::value));
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<ClipMaskIterator>::value_type,
         ClipMask>::value));
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<NoDataIterator>::value_type,
         NoData>::value));

    for(size_t i = 0; i < nrValues; ++i, ++argument1, ++result,
         ++clipMask, ++noData) {
      if(!clipMaskPolicy.mask(*clipMask)) {
        if(!_noDataPolicy.isNoData(*noData)) {
          if(!_domainPolicy.inDomain(*argument1, argument2)) {
            _noDataPolicy.setNoData(*noData);
          }
          else {
            assert(!boost::math::isnan(*argument1));
            assert(!boost::math::isnan(argument2));

            *result = _algorithm(*argument1, argument2);

            if(!_rangePolicy.inRange(*argument1, argument2, *result)) {
              _noDataPolicy.setNoData(*noData);
            }
          }
        }
      }
    }
  }

  //!
  /*!
    \tparam    .
    \param     .
    \return    .
    \exception .
    \warning   .
    \sa        .

    operation(scalar*, *scalar, scalar*)
  */
  template<
         class ArgumentIterator,
         class ResultIterator,
         class ClipMaskIterator,
         class NoDataIterator>
  inline void operator()(
         ArgumentIterator argument1,
         ArgumentIterator argument2,
         ResultIterator result,
         ClipMaskIterator clipMask,
         NoDataIterator noData,
         size_t nrValues) const
  {
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<ArgumentIterator>::value_type,
         T>::value));
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<ResultIterator>::value_type,
         T>::value));
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<ClipMaskIterator>::value_type,
         ClipMask>::value));
    BOOST_STATIC_ASSERT((boost::is_same<
         typename std::iterator_traits<NoDataIterator>::value_type,
         NoData>::value));

    for(size_t i = 0; i < nrValues; ++i, ++argument1, ++argument2, ++result,
         ++clipMask, ++noData) {
      if(!clipMaskPolicy.mask(*clipMask)) {
        if(!_noDataPolicy.isNoData(*noData)) {
          if(!_domainPolicy.inDomain(*argument1, *argument2)) {
            _noDataPolicy.setNoData(*noData);
          }
          else {
            assert(!boost::math::isnan(*argument1));
            assert(!boost::math::isnan(*argument2));

            *result = _algorithm(*argument1, *argument2);

            if(!_rangePolicy.inRange(*argument1, *argument2, *result)) {
              _noDataPolicy.setNoData(*noData);
            }
          }
        }
      }
    }
  }
};

} // namespace framework
} // namespace binary
} // namespace operations
} // namespace ranally

#endif