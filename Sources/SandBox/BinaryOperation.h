#ifndef INCLUDED_OPERATIONPOLICIES
#include "OperationPolicies.h"
#define INCLUDED_OPERATIONPOLICIES
#endif



//! Class for binary operations.
/*!
  \tparam    Argument Type of argument values.
  \tparam    Result Type of result values.
  \tparam    Operation Concrete binary operation.
  \tparam    NoDataHandler Handler for missing value.
  \tparam    ValueDomainHandler Handler checking domain restrictions.
  \tparam    MaskHandler Handler for masking.
  \sa        UnaryOperation, NAryOperation

  The operation is used to calculate new values. A default constructed object
  is layered within the class.

  This class has implementations of operator() for different permutations of
  array - constant arguments:
  - array, array
  - array, constant
  - constant, array
  - constant, constant
*/
template<class Argument, class Result,
         template<typename, typename> class Operation,
         class NoDataHandler=IgnoreNoData<Result>,
         class ValueDomainHandler=DontCheckValueDomain<Argument>,
         class MaskHandler=DontMask>
struct BinaryOperation: public NoDataHandler,
                        public ValueDomainHandler,
                        public MaskHandler
{
  Operation<Argument, Result> operation;

  template<class InputIterator, class OutputIterator>
  void operator()(
         OutputIterator destination,
         InputIterator source1,
         InputIterator source2,
         size_t nrValues)
  {
    for(size_t i = 0; i < nrValues; ++i) {
      if(!this->mask(i)) {
        operator()(*destination, *source1, *source2);
      }

      ++destination;
      ++source1;
      ++source2;
    }
  }

  template<class InputIterator, class OutputIterator>
  void operator()(
         OutputIterator destination,
         InputIterator source1,
         Argument source2,
         size_t nrValues)
  {
    for(size_t i = 0; i < nrValues; ++i) {
      if(!this->mask(i)) {
        operator()(*destination, *source1, source2);
      }

      ++destination;
      ++source1;
    }
  }

  template<class InputIterator, class OutputIterator>
  void operator()(
         OutputIterator destination,
         Argument source1,
         InputIterator source2,
         size_t nrValues)
  {
    for(size_t i = 0; i < nrValues; ++i) {
      if(!this->mask(i)) {
        operator()(*destination, source1, *source2);
      }

      ++destination;
      ++source2;
    }
  }

  inline void operator()(
         Result& destination,
         Argument source1,
         Argument source2)
  {
    if(!inDomain(source1) || !inDomain(source2)) {
      setNoData(destination);
    }
    else {
      operation(destination, source1, source2);
    }
  }

  inline void operator()(
         std::vector<bool>::reference destination,
         Argument source1,
         Argument source2)
  {
    if(!inDomain(source1) || !inDomain(source2)) {
      this->setNoData(destination);
    }
    else {
      operation(destination, source1, source2);
    }
  }
};


