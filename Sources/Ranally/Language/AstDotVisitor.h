#ifndef INCLUDED_RANALLY_ASTDOTVISITOR
#define INCLUDED_RANALLY_ASTDOTVISITOR

#include "Ranally/Language/DotVisitor.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AstDotVisitor:
  public DotVisitor
{

  friend class AstDotVisitorTest;

public:

  enum Mode {
    Declaring=0x1,
    ConnectingAst=0x2,
    ConnectingCfg=0x4,
    ConnectingUses=0x8
  };

                   AstDotVisitor       (int modes=0);

  /* virtual */    ~AstDotVisitor      ();

  void             Visit               (language::AssignmentVertex& vertex);

  void             Visit               (language::FunctionVertex& vertex);

  void             Visit               (language::IfVertex& vertex);

  void             Visit               (language::NameVertex& vertex);

  void             Visit               (language::OperatorVertex& vertex);

  void             Visit               (language::ScriptVertex& vertex);

  void             Visit               (language::StringVertex& vertex);

  void             Visit               (language::WhileVertex& vertex);

protected:

private:

  //! Current mode.
  Mode             _mode;

  //! Modes to process.
  int              _modes;

  void             setMode             (Mode mode);

  void             addAstVertex        (
                                  language::SyntaxVertex const& sourceVertex,
                                  language::SyntaxVertex const& targetVertex);

  void             addCfgVertices      (
                                  language::SyntaxVertex const& sourceVertex);

  void             addUseVertices      (
                                  language::NameVertex const& vertex);

};

} // namespace ranally

#endif
