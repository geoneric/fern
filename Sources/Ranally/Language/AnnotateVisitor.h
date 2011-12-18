#ifndef INCLUDED_RANALLY_LANGUAGE_ANNOTATEVISITOR
#define INCLUDED_RANALLY_LANGUAGE_ANNOTATEVISITOR

#include "Ranally/Operation/Operations.h"
#include "Ranally/Language/Visitor.h"



namespace ranally {
namespace language {

//! Class for visitors that annotate the syntax tree.
/*!
  For each operation in the tree, the requirements are looked up. Some of these
  are stored in the operation vertex (eg: number of results, number of
  arguments) and others are stored in the expression vertices (eg: data type,
  value type).

  Apart from determining whether each operation exists or not, no validation
  is performed by this visitor. It only annotates the tree. Use a
  ValidateVisitor to perform the actual validation.

  For example, in case of a FunctionVertex, it is no problem if the operation
  is not known. Annotation is optional. The ValidateVisitor will check if all
  information required for execution is present.

  \sa        ValidateVisitor
*/
class AnnotateVisitor:
  public Visitor
{

  friend class AnnotateVisitorTest;

public:

                   AnnotateVisitor     (
                                  operation::OperationsPtr const& operations);

                   ~AnnotateVisitor    ();

private:

  operation::OperationsPtr _operations;

  void             Visit               (AssignmentVertex& vertex);

  void             Visit               (FunctionVertex& vertex);

  void             Visit               (IfVertex& vertex);

  void             Visit               (NameVertex& vertex);

  void             Visit               (OperatorVertex& vertex);

  void             Visit               (ScriptVertex& vertex);

  void             Visit               (WhileVertex& vertex);

};

} // namespace language
} // namespace ranally

#endif
