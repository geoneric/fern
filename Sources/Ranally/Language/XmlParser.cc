#include "Ranally/Language/XmlParser.h"
#include <sstream>
#include <stack>
#include <boost/make_shared.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/Language/FunctionVertex.h"
#include "Ranally/Language/IfVertex.h"
#include "Ranally/Language/NameVertex.h"
#include "Ranally/Language/NumberVertex.h"
#include "Ranally/Language/OperatorVertex.h"
#include "Ranally/Language/Ranally-pskel.hxx"
#include "Ranally/Language/StringVertex.h"
#include "Ranally/Language/SyntaxVertex.h"
#include "Ranally/Language/WhileVertex.h"


namespace {

class Ranally_pimpl:
    public ranally::Ranally_pskel
{

public:

    typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
        StatementVertices;

    void pre()
    {
        _statementVertices.clear();
    }

    void source(
        std::string const& sourceName)
    {
        _sourceName = ranally::String(sourceName);
    }

    void Statements(
        StatementVertices const& vertices)
    {
        assert(_statementVertices.empty());
        _statementVertices = vertices;
    }

    boost::shared_ptr<ranally::ScriptVertex> post_Ranally()
    {
        return boost::make_shared<ranally::ScriptVertex>(_sourceName,
            _statementVertices);
    }

private:

    ranally::String  _sourceName;

    StatementVertices _statementVertices;

};


class Assignment_pimpl:
    public ranally::Assignment_pskel
{

public:

    void pre()
    {
        _expressions.clear();
    }

    // void Targets(
    //   std::vector<boost::shared_ptr<ranally::ExpressionVertex> > const&
    //     vertices)
    // {
    //   assert(!vertices.empty());
    //   _targets = vertices;
    // }

    void Expression(
        boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        _expressions.push_back(vertex);
        assert(_expressions.size() <= 2);
    }

    boost::shared_ptr<ranally::AssignmentVertex> post_Assignment()
    {
        assert(!_expressions.empty());
        return boost::make_shared<ranally::AssignmentVertex>(
            _expressions[0], _expressions[1]);
    }

private:

    std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
        _expressions;

};


// class Targets_pimpl: public ranally::Targets_pskel
// {
// private:
//   std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
//     _vertices;
// 
// public:
//   void pre()
//   {
//     _vertices.clear();
//   }
// 
//   void Expression(
//     boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
//   {
//     assert(vertex);
//     _vertices.push_back(vertex);
//   }
// 
//   std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
//     post_Targets()
//   {
//     assert(!_vertices.empty());
//     return _vertices;
//   }
// };


class If_pimpl: public ranally::If_pskel
{

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
        std::vector<boost::shared_ptr<ranally::StatementVertex> >
            const& vertices)
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
        return boost::make_shared<ranally::IfVertex>(
            result.conditionVertex, result.trueStatementVertices,
            result.falseStatementVertices);
    }

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

};


class While_pimpl:
    public ranally::While_pskel
{

public:

    void pre()
    {
        _dataStack.push(WhileData());
    }

    void Expression(
        boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_dataStack.top().conditionVertex);
        assert(vertex);
        _dataStack.top().conditionVertex = vertex;
    }

    void Statements(
        std::vector<boost::shared_ptr<ranally::StatementVertex> >
            const& vertices)
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

    boost::shared_ptr<ranally::WhileVertex> post_While()
    {
        assert(!_dataStack.empty());
        WhileData result(_dataStack.top());
        _dataStack.pop();
        return boost::make_shared<ranally::WhileVertex>(
            result.conditionVertex, result.trueStatementVertices,
                result.falseStatementVertices);
    }

private:

    typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
        StatementVertices;

    struct WhileData
    {
        boost::shared_ptr<ranally::ExpressionVertex> conditionVertex;
        StatementVertices trueStatementVertices;
        StatementVertices falseStatementVertices;
    };

    std::stack<WhileData> _dataStack;

};


class Statements_pimpl:
    public ranally::Statements_pskel
{

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

    std::vector<boost::shared_ptr<ranally::StatementVertex> >
        post_Statements()
    {
        assert(!_dataStack.empty());
        StatementsData result(_dataStack.top());
        _dataStack.pop();
        return result;
    }

private:

    typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
        StatementsData;

    std::stack<StatementsData> _dataStack;

};


class Statement_pimpl:
    public ranally::Statement_pskel
{

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

    void While(
        boost::shared_ptr<ranally::WhileVertex> const& vertex)
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

private:

    typedef boost::shared_ptr<ranally::StatementVertex> StatementData;

    std::stack<StatementData> _dataStack;

};


class Expressions_pimpl:
    public ranally::Expressions_pskel
{

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

    std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
        post_Expressions()
    {
        assert(!_dataStack.empty());
        ExpressionsData result(_dataStack.top());
        _dataStack.pop();
        return result;
    }

private:

    typedef std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
        ExpressionsData;

    std::stack<ExpressionsData> _dataStack;

};


class Integer_pimpl:
    public ranally::Integer_pskel
{

public:

    void pre()
    {
    }

    // TODO correct type? Can be smaller than this.
    void Size(
        unsigned long long size)
    {
        _size = size;
    }

    void Value(
        // TODO correct type? Should be larger than this (long long).
        int value)
    {
        _value = value;
    }

    boost::shared_ptr<ranally::ExpressionVertex> post_Integer()
    {
        boost::shared_ptr<ranally::ExpressionVertex> result;

        switch(_size) {
            case 8: {
                result = boost::make_shared<ranally::NumberVertex<int8_t> >(
                    _value);
                break;
            }
            case 16: {
                result = boost::make_shared<ranally::NumberVertex<int16_t> >(
                    _value);
                break;
            }
            case 32: {
                result = boost::make_shared<ranally::NumberVertex<int32_t> >(
                    _value);
                break;
            }
            case 64: {
                result = boost::make_shared<ranally::NumberVertex<int64_t> >(
                    _value);
                break;
            }
            default: {
                assert(false);
                // TODO raise exception
                break;
            }
        }

        return result;
    }

private:

  unsigned long long _size;

  long long        _value;

};


class UnsignedInteger_pimpl:
    public ranally::UnsignedInteger_pskel
{

public:

    void pre()
    {
    }

    // TODO correct type? Can be smaller than this.
    void Size(
        unsigned long long size)
    {
        _size = size;
    }

    void Value(
        unsigned long long value)
    {
        _value = value;
    }

    boost::shared_ptr<ranally::ExpressionVertex>
        post_UnsignedInteger()
    {
        boost::shared_ptr<ranally::ExpressionVertex> result;

        switch(_size) {
            case 8: {
                result = boost::make_shared<ranally::NumberVertex<uint8_t> >(
                    _value);
                break;
            }
            case 16: {
                result = boost::make_shared<ranally::NumberVertex<uint16_t> >(
                    _value);
                break;
            }
            case 32: {
                result = boost::make_shared<ranally::NumberVertex<uint32_t> >(
                    _value);
                break;
            }
            case 64: {
                result = boost::make_shared<ranally::NumberVertex<uint64_t> >(
                    _value);
                break;
            }
            default: {
                assert(false);
                // TODO raise exception
                break;
            }
        }

        return result;
    }

private:

    unsigned long long _size;

    unsigned long long _value;

};



class Float_pimpl:
    public ranally::Float_pskel
{

public:

    void pre()
    {
    }

    // TODO correct type? Can be smaller than this.
    void Size(
        unsigned long long size)
    {
        _size = size;
    }

    void Value(
        double value)
    {
        _value = value;
    }

    boost::shared_ptr<ranally::ExpressionVertex> post_Float()
    {
        boost::shared_ptr<ranally::ExpressionVertex> result;

        switch(_size) {
            case 32: {
                assert(sizeof(float) == 4);
                result = boost::make_shared<ranally::NumberVertex<float> >(
                    _value);
                break;
            }
            case 64: {
                assert(sizeof(double) == 8);
                result = boost::make_shared<ranally::NumberVertex<double> >(
                    _value);
                break;
            }
            default: {
                assert(false);
                // TODO raise exception
                break;
            }
        }

        return result;
    }

private:

  unsigned long long _size;

  double           _value;

};


class Number_pimpl:
    public ranally::Number_pskel
{

public:

    void pre()
    {
        _vertex.reset();
    }

    void Integer(
        boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_vertex);
        _vertex = vertex;
    }

    void UnsignedInteger(
        boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_vertex);
        _vertex = vertex;
    }

    void Float(
        boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_vertex);
        _vertex = vertex;
    }

    boost::shared_ptr<ranally::ExpressionVertex> post_Number()
    {
        assert(_vertex);
        return _vertex;
    }

private:

    boost::shared_ptr<ranally::ExpressionVertex> _vertex;

};


class Function_pimpl:
    public ranally::Function_pskel
{

public:

    void pre()
    {
        _dataStack.push(FunctionData());
    }

    void Name(
        std::string const& name)
    {
        assert(!_dataStack.empty());
        _dataStack.top().name = ranally::String(name);
    }

    void Expressions(
        std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
            const& vertices)
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
        return boost::make_shared<ranally::FunctionVertex>(
            result.name, result.expressionVertices);
    }

private:

    typedef std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
        ExpressionVertices;

    struct FunctionData
    {
        ranally::String name;
        ExpressionVertices expressionVertices;
    };

    std::stack<FunctionData> _dataStack;

};


class Operator_pimpl:
    public ranally::Operator_pskel
{

public:

    void pre()
    {
        _dataStack.push(OperatorData());
    }

    void Name(
        std::string const& name)
    {
        assert(!_dataStack.empty());
        _dataStack.top().name = ranally::String(name);
    }

    void Expressions(
        std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
            const& vertices)
    {
        assert(!_dataStack.empty());
        assert(_dataStack.top().expressionVertices.empty());
        _dataStack.top().expressionVertices = vertices;
    }

    boost::shared_ptr<ranally::OperatorVertex> post_Operator()
    {
        assert(!_dataStack.empty());
        OperatorData result(_dataStack.top());
        _dataStack.pop();
        return boost::make_shared<ranally::OperatorVertex>(
            result.name, result.expressionVertices);
    }

private:

    typedef std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
        ExpressionVertices;

    struct OperatorData
    {
        ranally::String name;
        ExpressionVertices expressionVertices;
    };

    std::stack<OperatorData> _dataStack;

};


class Expression_pimpl:
    public ranally::Expression_pskel
{

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
        _dataStack.top().vertex = boost::make_shared<
            ranally::NameVertex>(_dataStack.top().line, _dataStack.top().col,
            ranally::String(name));
    }

    void String(
        std::string const& string)
    {
        assert(!_dataStack.empty());
        assert(!_dataStack.top().vertex);
        _dataStack.top().vertex = boost::make_shared<
            ranally::StringVertex>(_dataStack.top().line,
                _dataStack.top().col, ranally::String(string));
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

    void Operator(
        boost::shared_ptr<ranally::OperatorVertex> const& vertex)
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

private:

  struct ExpressionData
  {
      int line;
      int col;
      boost::shared_ptr<ranally::ExpressionVertex> vertex;
  };

  std::stack<ExpressionData> _dataStack;

};

} // Anonymous namespace


namespace ranally {

//! Constructor.
/*!
*/
XmlParser::XmlParser()
{
}


//! Destructor.
/*!
*/
XmlParser::~XmlParser()
{
}


//! Parse the Xml in \a stream and return a syntax tree.
/*!
  \param     stream Stream with Xml to parse.
  \return    Root of syntax tree.
  \exception std::exception In case of a System category error.
  \exception xml_schema::parsing In case of Xml category error.
  \warning   .
  \sa        .
*/
boost::shared_ptr<ScriptVertex> XmlParser::parse(
    std::istream& stream) const
{
    // Python stores regular integers in a C long. Xsd doesn't have a long
    // parser, but does have an int parser. Let's make sure a long is of the
    // same size as an int. Xsd's long parser uses long long, which is good
    // for Pythons long integer type.
    // TODO Use long_p for integer parsing? int is too small I would think.
    // assert(sizeof(int) == sizeof(long));

    xml_schema::positive_integer_pimpl positive_integer_p;
    xml_schema::int_pimpl int_p;
    xml_schema::non_negative_integer_pimpl non_negative_integer_p;
    xml_schema::long_pimpl long_p;
    xml_schema::double_pimpl double_p;
    xml_schema::string_pimpl string_p;

    Integer_pimpl integer_p;
    integer_p.parsers(positive_integer_p, int_p);

    UnsignedInteger_pimpl unsigned_integer_p;
    unsigned_integer_p.parsers(positive_integer_p, non_negative_integer_p);

    Float_pimpl float_p;
    float_p.parsers(positive_integer_p, double_p);

    Number_pimpl number_p;
    number_p.parsers(integer_p, unsigned_integer_p, float_p);

    Expression_pimpl expression_p;
    // TODO Set parsers?

    Expressions_pimpl expressions_p;
    expressions_p.parsers(expression_p);

    Function_pimpl function_p;
    function_p.parsers(string_p, expressions_p);

    Operator_pimpl operator_p;
    operator_p.parsers(string_p, expressions_p);

    expression_p.parsers(string_p, string_p, number_p, function_p, operator_p,
      non_negative_integer_p, non_negative_integer_p);

    // Targets_pimpl targets_p;
    // targets_p.parsers(expression_p);

    Assignment_pimpl assignment_p;
    assignment_p.parsers(expression_p);

    Statements_pimpl statements_p;

    If_pimpl if_p;
    if_p.parsers(expression_p, statements_p);

    While_pimpl while_p;
    while_p.parsers(expression_p, statements_p);

    Statement_pimpl statement_p;
    statement_p.parsers(expression_p, assignment_p, if_p, while_p);

    statements_p.parsers(statement_p);

    Ranally_pimpl ranally_p;
    ranally_p.parsers(statements_p, string_p);

    xml_schema::document doc_p(ranally_p, "Ranally");

    ranally_p.pre();
    doc_p.parse(stream);
    return ranally_p.post_Ranally();
}


/*!
  \overload
  \param     xml String with Xml to parse.
*/
boost::shared_ptr<ScriptVertex> XmlParser::parse(
    String const& xml) const
{
    // Copy string contents in a string stream and work with that.
    std::stringstream stream;
    stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    stream << xml.encodeInUTF8(); // << std::endl;

    return parse(stream);
}

} // namespace ranally
