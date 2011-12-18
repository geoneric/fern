#ifndef INCLUDED_RANALLY_LANGUAGE_OPERATIONVERTEX
#define INCLUDED_RANALLY_LANGUAGE_OPERATIONVERTEX

#include "Ranally/Operation/Operation.h"
#include "Ranally/Language/ExpressionVertex.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OperationVertex:
  public ExpressionVertex
{

  friend class OperationVertexTest;

public:

  virtual          ~OperationVertex    ();

  void             setOperation        (
                                  operation::OperationPtr const& operation);

  operation::OperationPtr const& operation() const;

protected:

                   OperationVertex     (UnicodeString const& name);

private:

  operation::OperationPtr _operation;

};

} // namespace language
} // namespace ranally

#endif
