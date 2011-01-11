#include "DotVisitor.h"

#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include "Vertices.h"



namespace ranally {

DotVisitor::DotVisitor(
  Type type)

  : _type(type),
    _mode(Declaring) /// ,
    /// _definition(0)

{
}



DotVisitor::~DotVisitor()
{
}



UnicodeString const& DotVisitor::script() const
{
  return _script;
}



void DotVisitor::addAstVertex(
  language::SyntaxVertex const& sourceVertex,
  language::SyntaxVertex const& targetVertex)
{
  assert(_mode == ConnectingAst);
  _script +=
    UnicodeString((boost::format("\"%1%\"") % &sourceVertex).str().c_str()) +
    " -> " +
    (boost::format("\"%1%\"") % &targetVertex).str().c_str() + " ["
    "];\n";
}



void DotVisitor::addCfgVertices(
  language::SyntaxVertex const& sourceVertex)
{
  assert(_mode == ConnectingCfg);
  BOOST_FOREACH(language::SyntaxVertex const* successor,
    sourceVertex.successors()) {
    _script +=
      UnicodeString((boost::format("\"%1%\"") % &sourceVertex).str().c_str()) +
      " -> " +
      (boost::format("\"%1%\"") % successor).str().c_str() + " ["
        "color=red, "
        "constraint=false, "
        "style=dashed"
      "];\n";
  }
}



void DotVisitor::addUseVertices(
  language::NameVertex const& vertex)
{
  assert(_mode == ConnectingUses);
  BOOST_FOREACH(language::NameVertex const* use, vertex.uses()) {
    _script +=
      UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
      " -> " +
      (boost::format("\"%1%\"") % use).str().c_str() + " ["
        "color=blue, "
        "constraint=false, "
        "style=dotted"
      "];\n";
  }
}



void DotVisitor::addFlowgraphVertex(
  language::SyntaxVertex const& /* sourceVertex */,
  language::SyntaxVertex const& /* targetVertex */)
{
  /// assert(_mode == ConnectingFlowgraph);

  /// // In case the operation argument is a NameVertex, we want to connect the
  /// // defining location to the target. Let's find the definition. This only
  /// // succeeds if the source vertex is a NameVertex and if it has a definition.
  /// // TODO Can't we depend on preprocessing (clean up) of the tree, instead of
  /// //      diving in? We may want to show the current state of the tree, whatever
  /// //      it is, instead of optimizing the plot by finding stuff ourselves.
  /// assert(!_definition);
  /// _mode = ConnectingOperationArgument;
  /// const_cast<language::SyntaxVertex&>(sourceVertex).Accept(*this);
  /// language::SyntaxVertex const* newSourceVertex = _definition
  ///   ? _definition
  ///   : &sourceVertex;

  /// _script +=
  ///   UnicodeString((boost::format("\"%1%\"") % newSourceVertex).str().c_str()) +
  ///   " -> " +
  ///   (boost::format("\"%1%\"") % &targetVertex).str().c_str() + " ["
  ///   "];\n";

  /// _mode = ConnectingFlowgraph;
  /// _definition = 0;
}



// void DotVisitor::addFlowgraphVertices(
//   NameVertex const& vertex)
// {
//   assert(_mode == ConnectingFlowgraph);
//   BOOST_FOREACH(NameVertex const* use, vertex.uses()) {
//     _script +=
//       UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
//       " -> " +
//       (boost::format("\"%1%\"") % use).str().c_str() + " ["
//       "];\n";
//   }
// }



void DotVisitor::Visit(
  language::AssignmentVertex& vertex)
{
  // assert(_mode != ConnectingOperationArgument);
  language::ExpressionVertices const& targets = vertex.targets();
  language::ExpressionVertices const& expressions = vertex.expressions();

  switch(_mode) {
    case Declaring: {
      if(_type != Flowgraph) {
        _script +=
          UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
          " [label=\"=\"];\n";
      }
      break;
    }
    case ConnectingAst: {
      assert(expressions.size() == targets.size());
      for(size_t i = 0; i < expressions.size(); ++i) {
        addAstVertex(vertex, *vertex.targets()[i]);
        addAstVertex(vertex, *vertex.expressions()[i]);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
    /// case ConnectingFlowgraph: {
    ///   assert(expressions.size() == targets.size());
    ///   for(size_t i = 0; i < expressions.size(); ++i) {
    ///     addFlowgraphVertex(*vertex.expressions()[i], *vertex.targets()[i]);
    ///   }

    ///   // hier verder
    ///   // a = b + c
    ///   // d = f(a)

    ///   // b -> +
    ///   // c -> +
    ///   // + -> f
    ///   //
    ///   // Prereq: For each name it must be known what the address is of the
    ///   //         defining expression.
    ///   // 1. defining expression of operands -> operator
    ///   // 2. defining expression of arguments -> functions
    ///   //
    ///   // Connect defining occurrence with use occurrence of identifiers.
    ///   //
    ///   // assert(expressions.size() == targets.size());
    ///   // for(size_t i = 0; i < expressions.size(); ++i) {
    ///   //   NameVertex* nameVertex = dynamic_cast<NameVertex*>(targets[i]);
    ///   //   assert(nameVertex);
    ///   // }

    ///   break;
    /// }
    /// case ConnectingOperationArgument: {
    ///   break;
    /// }
  }

  /// if(_mode != ConnectingOperationArgument) {
    BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
      expressionVertex, vertex.expressions()) {
      expressionVertex->Accept(*this);
    }

    BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
      expressionVertex, vertex.targets()) {
      expressionVertex->Accept(*this);
    }
  /// }
}



void DotVisitor::Visit(
  language::FunctionVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.name() + "\"];\n";
      break;
    }
    case ConnectingAst: {
      BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
        expressionVertex, vertex.expressions()) {
        addAstVertex(vertex, *expressionVertex);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
    /// case ConnectingFlowgraph: {
    ///   BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    ///     expressionVertex, vertex.expressions()) {
    ///     addFlowgraphVertex(*expressionVertex, vertex);
    ///   }
    ///   break;
    /// }
  }

  /// if(_mode != ConnectingOperationArgument) {
    BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
      expressionVertex, vertex.expressions()) {
      expressionVertex->Accept(*this);
    }
  /// }
}



void DotVisitor::Visit(
  language::OperatorVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      // TODO Implement symbol member.
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + vertex.symbol() + "\"];\n";
      break;
    }
    case ConnectingAst: {
      BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
        expressionVertex, vertex.expressions()) {
        addAstVertex(vertex, *expressionVertex);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
    /// case ConnectingFlowgraph: {
    ///   BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
    ///     expressionVertex, vertex.expressions()) {
    ///     addFlowgraphVertex(*expressionVertex, vertex);
    ///   }
    ///   break;
    /// }
  }

  /// if(_mode != ConnectingOperationArgument) {
    BOOST_FOREACH(boost::shared_ptr<language::ExpressionVertex>
      expressionVertex, vertex.expressions()) {
      expressionVertex->Accept(*this);
    }
  /// }
}



// void DotVisitor::Visit(
//   language::ScriptVertex& vertex)
// {
//   _script = UnicodeString((boost::format(
//     "digraph G {\n"
//     "rankdir=%1%;\n"
//   ) % (_type == Ast ? "BT" : "LR")).str().c_str());
// 
//   _mode = Declaring;
//   if(_type != Flowgraph) {
//     // TODO Use script name.
//     _script +=
//       UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
//       " [label=\"Script\"];\n";
//   }
//   BOOST_FOREACH(boost::shared_ptr<language::StatementVertex> statementVertex,
//     vertex.statements()) {
//     statementVertex->Accept(*this);
//   }
// 
//   switch(_type) {
//     case Ast: {
//       assert(false);
//       // _mode = ConnectingAst;
//       // BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
//       //   statementVertex, vertex.statements()) {
//       //   addAstVertex(vertex, *statementVertex);
//       //   statementVertex->Accept(*this);
//       // }
// 
//       // _mode = ConnectingCfg;
//       // addCfgVertices(vertex);
//       // BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
//       //   statementVertex, vertex.statements()) {
//       //   statementVertex->Accept(*this);
//       // }
// 
//       // _mode = ConnectingUses;
//       // BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
//       //   statementVertex, vertex.statements()) {
//       //   statementVertex->Accept(*this);
//       // }
// 
//       break;
//     }
//     case Flowgraph: {
//       _mode = ConnectingFlowgraph;
// 
//       BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
//         statementVertex, vertex.statements()) {
//         statementVertex->Accept(*this);
//       }
// 
//       break;
//     }
//   }
// 
//   _script += "}\n";
// }



void DotVisitor::Visit(
  language::StringVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"\\\"" + vertex.value() + "\\\"\", shape=box];\n";
      break;
    }
    case ConnectingAst: {
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
  }
}



void DotVisitor::Visit(
  language::NameVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      if(_type != Flowgraph || vertex.definition() == &vertex ||
        !vertex.definition()) {
        _script +=
          UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
          " [label=\"" + vertex.name() + "\"];\n";
      }
      break;
    }
    case ConnectingAst: {
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      addUseVertices(vertex);
      break;
    }
    /// case ConnectingFlowgraph: {
    ///   // addFlowgraphVertices(vertex);
    ///   break;
    /// }
    /// case ConnectingOperationArgument: {
    ///   assert(!_definition);
    ///   // The vertex.definition() can be 0.
    ///   _definition = vertex.definition();
    ///   break;
    /// }
  }
}



template<typename T>
void DotVisitor::Visit(
  language::NumberVertex<T>& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"" + (boost::format("%1%") % vertex.value()).str().c_str() +
        "\", shape=box];\n";
      break;
    }
    case ConnectingAst: {
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
  }
}



void DotVisitor::Visit(
  language::NumberVertex<int8_t>& vertex)
{
  return Visit<int8_t>(vertex);
}



void DotVisitor::Visit(
  language::NumberVertex<int16_t>& vertex)
{
  return Visit<int16_t>(vertex);
}



void DotVisitor::Visit(
  language::NumberVertex<int32_t>& vertex)
{
  return Visit<int32_t>(vertex);
}



void DotVisitor::Visit(
  language::NumberVertex<int64_t>& vertex)
{
  return Visit<int64_t>(vertex);
}



void DotVisitor::Visit(
  language::NumberVertex<uint8_t>& vertex)
{
  return Visit<uint8_t>(vertex);
}



void DotVisitor::Visit(
  language::NumberVertex<uint16_t>& vertex)
{
  return Visit<uint16_t>(vertex);
}



void DotVisitor::Visit(
  language::NumberVertex<uint32_t>& vertex)
{
  return Visit<uint32_t>(vertex);
}



void DotVisitor::Visit(
  language::NumberVertex<uint64_t>& vertex)
{
  return Visit<uint64_t>(vertex);
}



void DotVisitor::Visit(
  language::NumberVertex<float>& vertex)
{
  return Visit<float>(vertex);
}



void DotVisitor::Visit(
  language::NumberVertex<double>& vertex)
{
  return Visit<double>(vertex);
}



void DotVisitor::Visit(
  language::IfVertex& vertex)
{
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"If\", shape=diamond];\n";
      break;
    }
    case ConnectingAst: {
      addAstVertex(vertex, *vertex.condition());
      BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
        statementVertex, vertex.trueStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
        statementVertex, vertex.falseStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
    /// case ConnectingFlowgraph: {
    ///   addFlowgraphVertex(*vertex.condition(), vertex);

    ///   // TODO Turn these statements into a sub graph.
    ///   //      There is no flow graph connection to a certain expression, but
    ///   //      just to the sub graph and from the sub graph.

    ///   BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    ///     statementVertex, vertex.trueStatements()) {
    ///     addFlowgraphVertex(*statementVertex, vertex);
    ///   }
    ///   BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    ///     statementVertex, vertex.falseStatements()) {
    ///     addFlowgraphVertex(*statementVertex, vertex);
    ///   }

    ///   break;
    /// }
    // case ConnectingOperationArgument: {
    //   break;
    // }
  }

  vertex.condition()->Accept(*this);
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.trueStatements()) {
    statementVertex->Accept(*this);
  }
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.falseStatements()) {
    statementVertex->Accept(*this);
  }
}



void DotVisitor::Visit(
  language::WhileVertex& vertex)
{
  /// assert(_mode != ConnectingOperationArgument);
  switch(_mode) {
    case Declaring: {
      _script +=
        UnicodeString((boost::format("\"%1%\"") % &vertex).str().c_str()) +
        " [label=\"While\", shape=diamond];\n";
      break;
    }
    case ConnectingAst: {
      addAstVertex(vertex, *vertex.condition());
      BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
        statementVertex, vertex.trueStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
        statementVertex, vertex.falseStatements()) {
        addAstVertex(vertex, *statementVertex);
      }
      break;
    }
    case ConnectingCfg: {
      addCfgVertices(vertex);
      break;
    }
    case ConnectingUses: {
      break;
    }
    /// case ConnectingFlowgraph: {
    ///   addFlowgraphVertex(*vertex.condition(), vertex);
    ///   // BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    ///   //   statementVertex, vertex.trueStatements()) {
    ///   //   addFlowgraphVertex(*statementVertex, vertex);
    ///   // }
    ///   // BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    ///   //   statementVertex, vertex.falseStatements()) {
    ///   //   addFlowgraphVertex(*statementVertex, vertex);
    ///   // }
    ///   break;
    /// }
  }

  vertex.condition()->Accept(*this);
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.trueStatements()) {
    statementVertex->Accept(*this);
  }
  BOOST_FOREACH(boost::shared_ptr<language::StatementVertex>
    statementVertex, vertex.falseStatements()) {
    statementVertex->Accept(*this);
  }
}



void DotVisitor::setScript(
  UnicodeString const& string)
{
  _script = string;
}



void DotVisitor::addScript(
  UnicodeString const& string)
{
  _script += string;
}



void DotVisitor::setMode(
  Mode mode)
{
  _mode = mode;
}



DotVisitor::Mode DotVisitor::mode() const
{
  return _mode;
}

} // namespace ranally

