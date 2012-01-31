#ifndef INCLUDED_RANALLY_LANGUAGE_OPTIMIZEVISITOR
#define INCLUDED_RANALLY_LANGUAGE_OPTIMIZEVISITOR

#include <map>
#include "Ranally/Language/NameVertex.h"
#include "Ranally/Language/Visitor.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OptimizeVisitor
  : public Visitor
{

  friend class OptimizeVisitorTest;

public:

                   OptimizeVisitor     ();

                   ~OptimizeVisitor    ();

private:

  enum Mode {
    Defining,
    Using
  };

  Mode             _mode;

  std::map<ExpressionVertex const*, ExpressionVertexPtr> _inlineExpressions;

  std::vector<ExpressionVertexPtr> _inlinedExpressions;

  std::vector<StatementVertex*> _superfluousStatements;

  void             registerExpressionForInlining(
                                        ExpressionVertex const* use,
                                        ExpressionVertexPtr const& expression);

  void             visitStatements     (StatementVertices& statements);

  void             Visit               (AssignmentVertex& vertex);

  void             Visit               (NameVertex& vertex);

  void             Visit               (ScriptVertex& vertex);

};

} // namespace language
} // namespace ranally

#endif
