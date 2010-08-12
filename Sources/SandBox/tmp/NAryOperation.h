#ifndef INCLUDED_IOSTREAM
#include <iostream>
#define INCLUDED_IOSTREAM
#endif

#ifndef INCLUDED_OPERATIONPOLICIES
#include "OperationPolicies.h"
#define INCLUDED_OPERATIONPOLICIES
#endif



//! Class for n-ary operations.
/*!
  \tparam    Argument Type of argument values.
  \tparam    Result Type of result values.
  \tparam    Operation Concrete binary operation.
  \tparam    MissingValueHandler Handler for missing value.
  \tparam    ValueDomainHandler Handler checking domain restrictions.
  \tparam    MaskHandler Handler for masking.
  \sa        UnaryOperation, BinaryOperation
*/
template<class Argument, class Result,
         template<typename, typename> class Operation,
         class MissingValueHandler=DontHandleMissingValues<Result>,
         class ValueDomainHandler=DontCheckValueDomain<Argument>,
         class MaskHandler=DontMask>
struct NAryOperation: public MissingValueHandler,
                      public ValueDomainHandler,
                      public MaskHandler
{
  Operation<Argument, Result> operation;

  template<class InputIterator, class OutputIterator>
  void operator()(
         OutputIterator destination,
         InputIterator sources,
         size_t const nrCollections,
         size_t const nrValues)
  {
    assert(nrCollections > 0);

    // Loop over all values.
    for(size_t i = 0; i < nrValues; ++i) {
      if(!this->mask(i)) {
        if(!inDomain(**sources)) {
          setNoData(*destination);
          for(size_t j = 0; j < nrCollections; ++j, ++(*sources), ++sources) { }
        }
        else {
          // Initialize the result with the first source value from the first
          // collection.
          *destination = **sources;

          // Increase the iterator to the next source value in the first
          // collection.
          ++(*sources);

          // Increase the iterator to the next collection.
          ++sources;

          for(size_t j = 1; j < nrCollections; ++j, ++(*sources), ++sources) {
            if(!inDomain(**sources)) {
              setNoData(*destination);
              for(; j < nrCollections; ++j, ++(*sources), ++sources) { }
              break;
            }

            operation(*destination, *destination, **sources);
          }
        }

        // Reset the iterator to point to the first collection again.
        sources -= nrCollections;
      }

      ++destination;
    }
  }
};

