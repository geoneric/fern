#include "XmlParser.h"

#include <sstream>
#include <stack>
#include <boost/make_shared.hpp>

#include "dev_UnicodeUtils.h"

#include "Ranally-pskel.hxx"

#include "FunctionVertex.h"
#include "IfVertex.h"
#include "NameVertex.h"
#include "NumberVertex.h"
#include "StringVertex.h"
#include "SyntaxVertex.h"



namespace {

class Ranally_pimpl: public ranally::Ranally_pskel
{
private:

  typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
    StatementVertices;

  StatementVertices _statementVertices;

public:
  void pre()
  {
    _statementVertices.clear();
  }

  void Statements(
    std::vector<boost::shared_ptr<ranally::StatementVertex> > const& vertices)
  {
    assert(_statementVertices.empty());
    _statementVertices = vertices;
  }

  boost::shared_ptr<ranally::SyntaxTree> post_Ranally()
  {
    return boost::make_shared<ranally::SyntaxTree>(_statementVertices);
  }
};



class Assignment_pimpl: public ranally::Assignment_pskel
{
private:
  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > _targets;
  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > _expressions;

public:
  void pre()
  {
    _targets.clear();
    _expressions.clear();
  }

  void Targets(
    std::vector<boost::shared_ptr<ranally::ExpressionVertex> > const& vertices)
  {
    assert(!vertices.empty());
    _targets = vertices;
  }

  void Expressions(
    std::vector<boost::shared_ptr<ranally::ExpressionVertex> > const& vertices)
  {
    assert(!vertices.empty());
    _expressions = vertices;
  }

  boost::shared_ptr<ranally::AssignmentVertex> post_Assignment()
  {
    assert(!_targets.empty() && !_expressions.empty());
    return boost::make_shared<ranally::AssignmentVertex>(_targets,
      _expressions);
  }
};



class Targets_pimpl: public ranally::Targets_pskel
{
private:
  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > _vertices;

public:
  void pre()
  {
    _vertices.clear();
  }

  void Expression(
    boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
  {
    assert(vertex);
    _vertices.push_back(vertex);
  }

  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > post_Targets()
  {
    assert(!_vertices.empty());
    return _vertices;
  }
};



class If_pimpl: public ranally::If_pskel
{
private:
  typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
    StatementVertices;

  struct IfData
  {
    boost::shared_ptr<ranally::ExpressionVertex> conditionVertex;
    StatementVertices trueStatementVertices;
    StatementVertices falseStatementVertices;
  };

  std::stack<IfData> _dataStack;

public:
  void pre()
  {
    _dataStack.push(IfData());
  }

  void Expression(
    boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
  {
    assert(!_dataStack.top().conditionVertex);
    assert(vertex);
    _dataStack.top().conditionVertex = vertex;
  }

  void Statements(
    std::vector<boost::shared_ptr<ranally::StatementVertex> > const& vertices)
  {
    if(_dataStack.top().trueStatementVertices.empty()) {
      assert(!vertices.empty());
      _dataStack.top().trueStatementVertices = vertices;
    }
    else {
      assert(_dataStack.top().falseStatementVertices.empty());
      _dataStack.top().falseStatementVertices = vertices;
    }
  }

  boost::shared_ptr<ranally::IfVertex> post_If()
  {
    assert(!_dataStack.empty());
    IfData result(_dataStack.top());
    _dataStack.pop();
    return boost::make_shared<ranally::IfVertex>(result.conditionVertex,
      result.trueStatementVertices, result.falseStatementVertices);
  }
};



class Statements_pimpl: public ranally::Statements_pskel
{
private:
  typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
    StatementsData;

  std::stack<StatementsData> _dataStack;

public:
  void pre()
  {
    _dataStack.push(StatementsData());
  }

  void Statement(
    boost::shared_ptr<ranally::StatementVertex> const& vertex)
  {
    assert(vertex);
    _dataStack.top().push_back(vertex);
  }

  std::vector<boost::shared_ptr<ranally::StatementVertex> > post_Statements()
  {
    assert(!_dataStack.empty());
    StatementsData result(_dataStack.top());
    _dataStack.pop();
    return result;
  }
};



class Statement_pimpl: public ranally::Statement_pskel
{
private:
  typedef boost::shared_ptr<ranally::StatementVertex> StatementData;

  std::stack<StatementData> _dataStack;

public:
  void pre()
  {
    _dataStack.push(StatementData());
  }

  void Expression(
    boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
  {
    assert(vertex);
    assert(!_dataStack.empty());
    _dataStack.top() = vertex;
  }

  void Assignment(
    boost::shared_ptr<ranally::AssignmentVertex> const& vertex)
  {
    assert(vertex);
    assert(!_dataStack.empty());
    _dataStack.top() = vertex;
  }

  void If(
    boost::shared_ptr<ranally::IfVertex> const& vertex)
  {
    assert(vertex);
    assert(!_dataStack.empty());
    _dataStack.top() = vertex;
  }

  boost::shared_ptr<ranally::StatementVertex> post_Statement()
  {
    assert(!_dataStack.empty());
    StatementData result(_dataStack.top());
    _dataStack.pop();
    return result;
  }
};



class Expressions_pimpl: public ranally::Expressions_pskel
{
private:
  typedef std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
    ExpressionsData;

  std::stack<ExpressionsData> _dataStack;

public:
  void pre()
  {
    _dataStack.push(ExpressionsData());
  }

  void Expression(
    boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
  {
    assert(vertex);
    _dataStack.top().push_back(vertex);
  }

  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > post_Expressions()
  {
    assert(!_dataStack.empty());
    ExpressionsData result(_dataStack.top());
    _dataStack.pop();
    return result;
  }
};



class Number_pimpl: public ranally::Number_pskel
{
private:
  boost::shared_ptr<ranally::ExpressionVertex> _vertex;

public:
  void pre()
  {
    _vertex.reset();
  }

  void Integer(
    int value)
  {
    assert(!_vertex);
    _vertex = boost::make_shared<ranally::NumberVertex<int> >(value);
  }

  void Long(
    long long value)
  {
    assert(!_vertex);
    _vertex = boost::make_shared<ranally::NumberVertex<long long> >(value);
  }

  void Double(
    double value)
  {
    assert(!_vertex);
    _vertex = boost::make_shared<ranally::NumberVertex<double> >(value);
  }

  boost::shared_ptr<ranally::ExpressionVertex> post_Number()
  {
    assert(_vertex);
    return _vertex;
  }
};



class Function_pimpl: public ranally::Function_pskel
{
private:
  typedef std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
    ExpressionVertices;

  struct FunctionData
  {
    UnicodeString name;
    ExpressionVertices expressionVertices;
  };

  std::stack<FunctionData> _dataStack;

public:
  void pre()
  {
    _dataStack.push(FunctionData());
  }

  void Name(
    std::string const& name)
  {
    assert(!_dataStack.empty());
    _dataStack.top().name = dev::decodeFromUTF8(name);
  }

  void Expressions(
    std::vector<boost::shared_ptr<ranally::ExpressionVertex> > const& vertices)
  {
    assert(!_dataStack.empty());
    assert(_dataStack.top().expressionVertices.empty());
    _dataStack.top().expressionVertices = vertices;
  }

  boost::shared_ptr<ranally::FunctionVertex> post_Function()
  {
    assert(!_dataStack.empty());
    FunctionData result(_dataStack.top());
    _dataStack.pop();
    return boost::make_shared<ranally::FunctionVertex>(result.name,
      result.expressionVertices);
  }
};



class Expression_pimpl: public ranally::Expression_pskel
{
private:
  struct ExpressionData
  {
    int line;
    int col;
    boost::shared_ptr<ranally::ExpressionVertex> vertex;
  };

  std::stack<ExpressionData> _dataStack;

public:
  void pre()
  {
    _dataStack.push(ExpressionData());
  }

  void line(
    unsigned long long line)
  {
    assert(!_dataStack.empty());
    _dataStack.top().line = line;
  }

  void col(
    unsigned long long col)
  {
    assert(!_dataStack.empty());
    _dataStack.top().col = col;
  }

  void Name(
    std::string const& name)
  {
    assert(!_dataStack.empty());
    assert(!_dataStack.top().vertex);
    _dataStack.top().vertex = boost::make_shared<ranally::NameVertex>(
      _dataStack.top().line, _dataStack.top().col, dev::decodeFromUTF8(name));
  }

  void String(
    std::string const& string)
  {
    assert(!_dataStack.empty());
    assert(!_dataStack.top().vertex);
    _dataStack.top().vertex = boost::make_shared<ranally::StringVertex>(
      _dataStack.top().line, _dataStack.top().col, dev::decodeFromUTF8(string));
  }

  void Number(
    boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
  {
    assert(!_dataStack.empty());
    assert(!_dataStack.top().vertex);
    _dataStack.top().vertex = vertex;
    _dataStack.top().vertex->setPosition(_dataStack.top().line,
      _dataStack.top().col);
  }

  void Function(
    boost::shared_ptr<ranally::FunctionVertex> const& vertex)
  {
    assert(!_dataStack.empty());
    assert(!_dataStack.top().vertex);
    _dataStack.top().vertex = vertex;
    _dataStack.top().vertex->setPosition(_dataStack.top().line,
      _dataStack.top().col);
  }

  boost::shared_ptr<ranally::ExpressionVertex> post_Expression()
  {
    assert(!_dataStack.empty());
    ExpressionData result(_dataStack.top());
    _dataStack.pop();
    return result.vertex;
  }
};

} // Anonymous namespace



namespace ranally {

XmlParser::XmlParser()
{
}



XmlParser::~XmlParser()
{
}



boost::shared_ptr<SyntaxTree> XmlParser::parse(
         std::istream& stream) const
{
  xml_schema::int_pimpl int_p;
  xml_schema::non_negative_integer_pimpl non_negative_integer_p;
  xml_schema::long_pimpl long_p;
  xml_schema::double_pimpl double_p;
  xml_schema::string_pimpl string_p;

  Number_pimpl number_p;
  number_p.parsers(int_p, long_p, double_p);

  Expression_pimpl expression_p;

  Expressions_pimpl expressions_p;
  expressions_p.parsers(expression_p);

  Function_pimpl function_p;
  function_p.parsers(string_p, expressions_p);

  expression_p.parsers(string_p, string_p, number_p, function_p,
    non_negative_integer_p, non_negative_integer_p);

  Targets_pimpl targets_p;
  targets_p.parsers(expression_p);

  Assignment_pimpl assignment_p;
  assignment_p.parsers(targets_p, expressions_p);

  Statements_pimpl statements_p;

  If_pimpl if_p;
  if_p.parsers(expression_p, statements_p /* , statements_p */);

  Statement_pimpl statement_p;
  statement_p.parsers(expression_p, assignment_p, if_p);

  statements_p.parsers(statement_p);

  Ranally_pimpl ranally_p;
  ranally_p.parsers(statements_p);

  xml_schema::document doc_p(ranally_p, "Ranally");

  ranally_p.pre();
  doc_p.parse(stream);
  return ranally_p.post_Ranally();
}



//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception std::exception In case of a System category error.
  \exception xml_schema::parsing In case of Xml category error.
  \warning   .
  \sa        .
*/
boost::shared_ptr<SyntaxTree> XmlParser::parse(
         UnicodeString const& xml) const
{
  // Copy string contents in a string stream and work with that.
  std::stringstream stream;
  stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
  stream << dev::encodeInUTF8(xml); // << std::endl;

  return parse(stream);
}

} // namespace ranally

