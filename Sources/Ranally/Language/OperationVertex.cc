#include "Ranally/Language/OperationVertex.h"



namespace ranally {
namespace language {

OperationVertex::OperationVertex(
  UnicodeString const& name)

  : ExpressionVertex(name)

{
}



OperationVertex::~OperationVertex()
{
}



boost::shared_ptr<operation::Requirements> const&
OperationVertex::requirements() const
{
  return _requirements;
}

} // namespace language
} // namespace ranally

