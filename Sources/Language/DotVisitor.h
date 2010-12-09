#ifndef INCLUDED_RANALLY_DOTVISITOR
#define INCLUDED_RANALLY_DOTVISITOR

#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>
#include <unicode/unistr.h>

#include "SyntaxVertex.h"



namespace ranally {

class AssignmentVertex;
class FunctionVertex;
class IfVertex;
class NameVertex;
template<typename T>
  class NumberVertex;
class OperatorVertex;
class ScriptVertex;
class StringVertex;
class WhileVertex;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  The dot graph is mainly useful for debugging purposes. The graph is handy
  for visualising the syntax-tree.
*/
class DotVisitor: private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<AssignmentVertex>,
  public Loki::Visitor<FunctionVertex>,
  public Loki::Visitor<IfVertex>,
  public Loki::Visitor<NameVertex>,
  public Loki::Visitor<NumberVertex<int8_t>>,
  public Loki::Visitor<NumberVertex<int16_t>>,
  public Loki::Visitor<NumberVertex<int32_t>>,
  public Loki::Visitor<NumberVertex<int64_t>>,
  public Loki::Visitor<NumberVertex<uint8_t>>,
  public Loki::Visitor<NumberVertex<uint16_t>>,
  public Loki::Visitor<NumberVertex<uint32_t>>,
  public Loki::Visitor<NumberVertex<uint64_t>>,
  public Loki::Visitor<NumberVertex<float>>,
  public Loki::Visitor<NumberVertex<double>>,
  public Loki::Visitor<OperatorVertex>,
  public Loki::Visitor<ScriptVertex>,
  public Loki::Visitor<StringVertex>,
  public Loki::Visitor<WhileVertex>
{

  friend class DotVisitorTest;

private:

  enum Type {
    Ast,
    Flowgraph
  };

  enum Mode {
    Declaring,
    ConnectingAst,
    ConnectingCfg,
    ConnectingUses,
    ConnectingFlowgraph,
    ConnectingOperationArgument
  };

  UnicodeString    _script;

  Type             _type;

  Mode             _mode;

  SyntaxVertex const* _definition;

  void             addAstVertex        (SyntaxVertex const& sourceVertex,
                                        SyntaxVertex const& targetVertex);

  void             addCfgVertices      (SyntaxVertex const& sourceVertex);

  void             addUseVertices      (NameVertex const& vertex);

  void             addFlowgraphVertex  (SyntaxVertex const& sourceVertex,
                                        SyntaxVertex const& targetVertex);

  // void             addFlowgraphVertices(NameVertex const& vertex);

  template<typename T>
  void             Visit               (NumberVertex<T>&);

protected:

public:

                   DotVisitor          ();

  /* virtual */    ~DotVisitor         ();

  UnicodeString const& script          () const;

  void             Visit               (AssignmentVertex&);

  void             Visit               (FunctionVertex&);

  void             Visit               (IfVertex&);

  void             Visit               (NameVertex&);

  void             Visit               (NumberVertex<int8_t>&);

  void             Visit               (NumberVertex<int16_t>&);

  void             Visit               (NumberVertex<int32_t>&);

  void             Visit               (NumberVertex<int64_t>&);

  void             Visit               (NumberVertex<uint8_t>&);

  void             Visit               (NumberVertex<uint16_t>&);

  void             Visit               (NumberVertex<uint32_t>&);

  void             Visit               (NumberVertex<uint64_t>&);

  void             Visit               (NumberVertex<float>&);

  void             Visit               (NumberVertex<double>&);

  void             Visit               (OperatorVertex&);

  void             Visit               (ScriptVertex&);

  void             Visit               (StringVertex&);

  void             Visit               (WhileVertex&);

};

} // namespace ranally

#endif
