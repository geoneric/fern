#ifndef INCLUDED_OPERATIONPOLICIES
#include "OperationPolicies.h"
#define INCLUDED_OPERATIONPOLICIES
#endif



//! Class for unary operations.
/*!
  \tparam    Argument Type of argument values.
  \tparam    Result Type of result values.
  \tparam    Operation Concrete binary operation.
  \tparam    NoDataHandler Handler for missing value.
  \tparam    ValueDomainHandler Handler checking domain restrictions.
  \tparam    MaskHandler Handler for masking.
  \sa        BinaryOperation, NAryOperation

  The operation is used to calculate new values. A default constructed object
  is layered within the class.

  This class has implementations of operator() for different permutations of
  array - constant arguments:
  - array
  - constant
*/
template<typename Argument, typename Result,
         template <typename, typename> class Operation,
         class NoDataHandler=IgnoreNoData<Result>,
         class ValueDomainHandler=DontCheckValueDomain<Argument>,
         class MaskHandler=DontMask>
struct UnaryOperation: public NoDataHandler,
                       public ValueDomainHandler,
                       public MaskHandler
{
  Operation<Argument, Result> operation;

  template<class InputIterator, class OutputIterator>
  void operator()(
         OutputIterator destination,
         InputIterator source,
         size_t nrValues)
  {
    for(size_t i = 0; i < nrValues; ++i) {
      if(!this->mask(i)) {
        operator()(*destination, *source);
      }

      ++destination;
      ++source;
    }
  }

  template<class OutputIterator>
  void operator()(
         OutputIterator destination,
         Argument source,
         size_t nrValues)
  {
    for(size_t i = 0; i < nrValues; ++i) {
      if(!this->mask(i)) {
        operator()(*destination, source);
      }

      ++destination;
    }
  }

  inline void operator()(
         Result& destination,
         Argument source)
  {
    if(!inDomain(source)) {
      setNoData(destination);
    }
    else {
      operation(destination, source);
    }
  }
};

