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

                   ~FlowgraphDotVisitor();

  void             Visit               (language::AssignmentVertex& vertex);

  void             Visit               (language::FunctionVertex& vertex);

  void             Visit               (language::IfVertex& vertex);

  void             Visit               (language::NameVertex& vertex);

  void             Visit               (language::NumberVertex<int8_t>& vertex);

  void             Visit               (language::NumberVertex<int16_t>& vertex);

  void             Visit               (language::NumberVertex<int32_t>& vertex);

  void             Visit               (language::NumberVertex<int64_t>& vertex);

  void             Visit               (language::NumberVertex<uint8_t>& vertex);

  void             Visit               (language::NumberVertex<uint16_t>& vertex);

  void             Visit               (language::NumberVertex<uint32_t>& vertex);

  void             Visit               (language::NumberVertex<uint64_t>& vertex);

  void             Visit               (language::NumberVertex<float>& vertex);

  void             Visit               (language::NumberVertex<double>& vertex);

  void             Visit               (language::OperatorVertex& vertex);

  void             Visit               (language::ScriptVertex& vertex);

  void             Visit               (language::StringVertex& vertex);

  void             Visit               (language::WhileVertex& vertex);

private:

  //! Current mode.
  Mode             _mode;

  void             setMode             (Mode mode);

  void             addFlowgraphVertex  (
                                  language::NameVertex const& sourceVertex,
                                  language::SyntaxVertex const& targetVertex);

  void             addFlowgraphVertex  (
                                  language::SyntaxVertex const& sourceVertex,
                                  language::SyntaxVertex const& targetVertex);

  // void             addFlowgraphVertices(NameVertex const& vertex);

  template<typename T>
  void             Visit               (language::NumberVertex<T>& vertex);

};

} // namespace ranally

#endif
