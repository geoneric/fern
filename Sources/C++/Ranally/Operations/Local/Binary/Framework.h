#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_FRAMEWORK
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_FRAMEWORK

namespace Ranally {
namespace Operations {
namespace Binary {
namespace Framework {

//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .

  This framework class is for binary local operations for which the argument
  and result types are the same.
*/
template<typename T,
         class Algorithm<T>
         class DomainPolicy,
         class RangePolicy,
         class NoDataPolicy
>
class BinarySame
{
private:

  Algorithm        _algorithm;

  DomainPolicy     _domainPolicy;

  RangePolicy      _rangePolicy;

  NoDataPolicy     _noDataPolicy;

public:

  DomainPolicy& domainPolicy() const
  {
    return _domainPolicy;
  }

  RangePolicy& rangePolicy() const
  {
    return _rangePolicy;
  }

  NoDataPolicy& noDataPolicy() const
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
         T& result) const
  {
    if(!_domainPolicy.inDomain(argument1, argument2)) {
      // At least one of the arguments is not in the operation's domain.
      _noDataPolicy.setNoData(result);
    }
    else {
      result = _algorithm(argument1, argument2);

      if(_rangePolicy.inRange(argument1, argument2, result)) {
        // The result is out of range.
        _noDataPolicy.setNoData(result);
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
  inline void operator()(
         T argument1,
         InputIterator argument2,
         OutputIterator result,
         size_t nrValues) const
  {
    for(size_t i = 0; i < nrValues; ++i, ++argument2, ++result) {
      if(!_domainPolicy.inDomain(argument1, *argument2)) {
        // At least one of the arguments is not in the operation's domain.
        _noDataPolicy.setNoData(*result);
      }
      else {
        *result = _algorithm(argument1, *argument2);

        if(_rangePolicy.inRange(argument1, *argument2, *result)) {
          // The result is out of range.
          _noDataPolicy.setNoData(*result);
        }
      }
    }
  }
};

} // namespace Framework
} // namespace Binary
} // namespace Operations
} // namespace Ranally

#endif
