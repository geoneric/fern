#include "Ranally/Language/OptimizeVisitor.h"
#include "Ranally/Language/AssignmentVertex.h"
#include "Ranally/Util/String.h"
#include <boost/range/algorithm/reverse.hpp>
#include <boost/range/algorithm/sort.hpp>



namespace ranally {
namespace language {

OptimizeVisitor::OptimizeVisitor()

  : Visitor()

{
}



OptimizeVisitor::~OptimizeVisitor()
{
}



void OptimizeVisitor::registerExpressionForInlining(
  ExpressionVertex const* use,
  ExpressionVertexPtr const& expression)
{
  _inlineExpressions[use] = expression;
}



void OptimizeVisitor::visitStatements(
  StatementVertices& statements)
{
  Visitor::visitStatements(statements);

  switch(_mode) {
    case Using: {
      break;
    }
    case Defining: {
      std::vector<size_t> statementsToErase, superfluousStatementsToErase;

      for(size_t i = 0; i < statements.size(); ++i) {
        for(size_t j = 0; j < _superfluousStatements.size(); ++j) {
          if(_superfluousStatements[j] == statements[i].get()) {
            statementsToErase.push_back(i);
            superfluousStatementsToErase.push_back(j);
          }
        }
      }

      boost::range::reverse(statementsToErase);
      for(size_t i = 0; i < statementsToErase.size(); ++i) {
        statements.erase(statements.begin() + statementsToErase[i]);
      }

      boost::range::sort(superfluousStatementsToErase);
      boost::range::reverse(superfluousStatementsToErase);
      for(size_t i = 0; i < superfluousStatementsToErase.size(); ++i) {
        _superfluousStatements.erase(_superfluousStatements.begin() +
          superfluousStatementsToErase[i]);
      }

      break;
    }
  }
}



void OptimizeVisitor::Visit(
  AssignmentVertex& vertex)
{
  // Inline the defining expression, if possible.
  switch(_mode) {
    case Using: {
      vertex.expression()->Accept(*this);

      std::map<ExpressionVertex const*, ExpressionVertexPtr>::iterator it =
        _inlineExpressions.find(vertex.expression().get());
      if(it != _inlineExpressions.end()) {
std::cout << "inserting " << ranally::util::encodeInUTF8((*it).second->name()) << std::endl;
        // Schedule the defining statement for removal.
        _inlinedExpressions.push_back((*it).second);
        vertex.setExpression((*it).second);
        _inlineExpressions.erase(it);
      }

      break;
    }
    case Defining: {
      vertex.target()->Accept(*this);

      std::vector<ExpressionVertexPtr>::iterator it = std::find(
        _inlinedExpressions.begin(), _inlinedExpressions.end(),
          vertex.expression());
      if(it != _inlinedExpressions.end()) {
        _superfluousStatements.push_back(&vertex);
        _inlinedExpressions.erase(it);
      }

      break;
    }
  }
}



void OptimizeVisitor::Visit(
  NameVertex& vertex)
{
  switch(_mode) {
    case Using: {
      std::vector<NameVertex*> const& definitions(vertex.definitions());

      if(definitions.size() == 1 && definitions[0]->uses().size() == 1) {
        // This identifier has one definition and the defining identifier is
        // used only here.
        assert(definitions[0]->uses()[0] == &vertex);

        // Register the value of the defining expression for inlining at the
        // use location.
std::cout << "register inlining of " << ranally::util::encodeInUTF8(vertex.name()) << " by " << ranally::util::encodeInUTF8(definitions[0]->value()->name()) << std::endl;
        registerExpressionForInlining(&vertex, definitions[0]->value());
      }

      break;
    }
    case Defining: {
      break;
    }
  }
}



void OptimizeVisitor::Visit(
  ScriptVertex& vertex)
{
  bool inlinedExpressions;

  do {
std::cout << "visit script" << std::endl;
    assert(_inlineExpressions.empty());
    assert(_inlinedExpressions.empty());
    assert(_superfluousStatements.empty());

    // First visit all use locations of name vertices.
    _mode = Using;
    Visitor::Visit(vertex);

    // If expressions have been inlined, then the script will change. We need
    // to repeat the visit until the script is stable.
    inlinedExpressions = !_inlinedExpressions.empty();

    // Now visit all defining locations of name vertices.
    _mode = Defining;
    Visitor::Visit(vertex);

    assert(_inlineExpressions.empty());
    assert(_inlinedExpressions.empty());
    assert(_superfluousStatements.empty());
  } while(inlinedExpressions);
}

} // namespace language
} // namespace ranally

