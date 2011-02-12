#ifndef INCLUDED_RANALLY_DOTVISITOR
#define INCLUDED_RANALLY_DOTVISITOR

#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>
#include <unicode/unistr.h>

#include "SyntaxVertex.h"



namespace ranally {

namespace language {

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

} // namespace language

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  The dot graph is mainly useful for debugging purposes. The graph is handy
  for visualising the syntax-tree.
*/
class DotVisitor: private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<language::AssignmentVertex>,
  public Loki::Visitor<language::FunctionVertex>,
  public Loki::Visitor<language::IfVertex>,
  public Loki::Visitor<language::NameVertex>,
  public Loki::Visitor<language::NumberVertex<int8_t>>,
  public Loki::Visitor<language::NumberVertex<int16_t>>,
  public Loki::Visitor<language::NumberVertex<int32_t>>,
  public Loki::Visitor<language::NumberVertex<int64_t>>,
  public Loki::Visitor<language::NumberVertex<uint8_t>>,
  public Loki::Visitor<language::NumberVertex<uint16_t>>,
  public Loki::Visitor<language::NumberVertex<uint32_t>>,
  public Loki::Visitor<language::NumberVertex<uint64_t>>,
  public Loki::Visitor<language::NumberVertex<float>>,
  public Loki::Visitor<language::NumberVertex<double>>,
  public Loki::Visitor<language::OperatorVertex>,
  public Loki::Visitor<language::ScriptVertex>,
  public Loki::Visitor<language::StringVertex>,
  public Loki::Visitor<language::WhileVertex>
{

  friend class DotVisitorTest;

protected:

  enum Type {
    Ast,
    Flowgraph
  };

  enum Mode {
    Declaring,
    ConnectingAst,
    ConnectingCfg,
    ConnectingUses /// ,
    /// ConnectingFlowgraph,
    /// ConnectingOperationArgument
  };

private:

  UnicodeString    _script;

  Type             _type;

  Mode             _mode;

  /// language::SyntaxVertex const* _definition;

  void             addUseVertices      (
                                  language::NameVertex const& vertex);

  void             addFlowgraphVertex  (
                                  language::SyntaxVertex const& sourceVertex,
                                  language::SyntaxVertex const& targetVertex);

  // void             addFlowgraphVertices(NameVertex const& vertex);

  template<typename T>
  void             Visit               (language::NumberVertex<T>&);

protected:

                   DotVisitor          (Type type);

  void             setScript           (UnicodeString const& string);

  void             addScript           (UnicodeString const& string);

  void             setMode             (Mode mode);

  Mode             mode                () const;

  void             addAstVertex        (
                                  language::SyntaxVertex const& sourceVertex,
                                  language::SyntaxVertex const& targetVertex);

  void             addCfgVertices      (
                                  language::SyntaxVertex const& sourceVertex);

public:

  /* virtual */    ~DotVisitor         ();

  UnicodeString const& script          () const;

  virtual void     Visit               (language::AssignmentVertex&);

  virtual void     Visit               (language::FunctionVertex&);

  virtual void     Visit               (language::IfVertex&);

  virtual void     Visit               (language::NameVertex&);

  virtual void     Visit               (language::NumberVertex<int8_t>&);

  virtual void     Visit               (language::NumberVertex<int16_t>&);

  virtual void     Visit               (language::NumberVertex<int32_t>&);

  virtual void     Visit               (language::NumberVertex<int64_t>&);

  virtual void     Visit               (language::NumberVertex<uint8_t>&);

  virtual void     Visit               (language::NumberVertex<uint16_t>&);

  virtual void     Visit               (language::NumberVertex<uint32_t>&);

  virtual void     Visit               (language::NumberVertex<uint64_t>&);

  virtual void     Visit               (language::NumberVertex<float>&);

  virtual void     Visit               (language::NumberVertex<double>&);

  virtual void     Visit               (language::OperatorVertex&);

  virtual void     Visit               (language::ScriptVertex& vertex)=0;

  virtual void     Visit               (language::StringVertex&);

  virtual void     Visit               (language::WhileVertex&);

};

} // namespace ranally

#endif
