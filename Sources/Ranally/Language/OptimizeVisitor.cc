#include "Ranally/Language/OptimizeVisitor.h"
#include "Ranally/Language/NameVertex.h"



namespace ranally {
namespace language {

OptimizeVisitor::OptimizeVisitor()

  : Visitor()

{
}



OptimizeVisitor::~OptimizeVisitor()
{
}



void OptimizeVisitor::Visit(
  NameVertex& vertex)
{
  std::vector<NameVertex*> const& uses(vertex.uses());

  if(uses.size() == 1) {
    // TODO
    // Directly connect the defining expression of this identifier with the
    // use location, and remove this vertex from the tree.
    // We cannot remove the vertex here. Maybe we can just disconnect it and
    // register it for deletion. Create a visitor for deleting orphaned
    // identifiers? In general, that may be a optimization task. Unused
    // identifiers can be part of the script too. This calls for different
    // optimization phases. Removing is the last phase.
  }
}

} // namespace language
} // namespace ranally

