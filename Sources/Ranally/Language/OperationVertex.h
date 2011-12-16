#ifndef INCLUDED_RANALLY_LANGUAGE_OPERATIONVERTEX
#define INCLUDED_RANALLY_LANGUAGE_OPERATIONVERTEX

#include "Ranally/Language/ExpressionVertex.h"
// #include "Ranally/Language/Operation/Requirements.h"



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

  // boost::shared_ptr<operation::Requirements> const& requirements() const;

protected:

                   OperationVertex     (UnicodeString const& name);

private:

  // boost::shared_ptr<operation::Requirements> _requirements;

};

} // namespace language
} // namespace ranally

#endif
