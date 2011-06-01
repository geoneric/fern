#ifndef INCLUDED_RANALLY_FLOWGRAPHDOTVISITOR
#define INCLUDED_RANALLY_FLOWGRAPHDOTVISITOR

#include "Ranally/Language/DotVisitor.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FlowgraphDotVisitor:
  public DotVisitor
{

  friend class FlowgraphDotVisitorTest;

public:

  enum Mode {
    Declaring=0x1,
    ConnectingFlowgraph=0x2
    /// ConnectingOperationArgument=0x4
  };

                   FlowgraphDotVisitor ();

  /* virtual */    ~FlowgraphDotVisitor();

  void             Visit               (language::ScriptVertex& vertex);

protected:

private:

  //! Current mode.
  Mode             _mode;

  void             setMode             (Mode mode);

  void             addFlowgraphVertex  (
                                  language::SyntaxVertex const& sourceVertex,
                                  language::SyntaxVertex const& targetVertex);

  // void             addFlowgraphVertices(NameVertex const& vertex);

};

} // namespace ranally

#endif
