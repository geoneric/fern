#include "ranally/ast/xml/xml_parser.h"
#include <sstream>
#include <stack>
#include "ranally/core/exception.h"
#include "ranally/ast/core/vertices.h"
#include "ranally/ast/xml/syntax_tree-pskel.hxx"


namespace {

class Ranally_pimpl:
    public ranally::Ranally_pskel
{

public:

    typedef std::vector<std::shared_ptr<ranally::StatementVertex>>
        StatementVertices;

    void pre()
    {
        _statement_vertices.clear();
    }

    void source(
        std::string const& sourceName)
    {
        _source_name = ranally::String(sourceName);
    }

    void Statements(
        StatementVertices const& vertices)
    {
        assert(_statement_vertices.empty());
        _statement_vertices = vertices;
    }

    std::shared_ptr<ranally::ScriptVertex> post_Ranally()
    {
        return std::shared_ptr<ranally::ScriptVertex>(new ranally::ScriptVertex(
            _source_name, _statement_vertices));
    }

private:

    ranally::String  _source_name;

    StatementVertices _statement_vertices;

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
    //   std::vector<std::shared_ptr<ranally::ExpressionVertex>> const&
    //     vertices)
    // {
    //   assert(!vertices.empty());
    //   _targets = vertices;
    // }

    void Expression(
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        _expressions.push_back(vertex);
        assert(_expressions.size() <= 2);
    }

    std::shared_ptr<ranally::AssignmentVertex> post_Assignment()
    {
        assert(!_expressions.empty());
        return std::shared_ptr<ranally::AssignmentVertex>(
            new ranally::AssignmentVertex(_expressions[0], _expressions[1]));
    }

private:

    std::vector<std::shared_ptr<ranally::ExpressionVertex>>
        _expressions;

};


// class Targets_pimpl: public ranally::Targets_pskel
// {
// private:
//   std::vector<std::shared_ptr<ranally::ExpressionVertex>>
//     _vertices;
// 
// public:
//   void pre()
//   {
//     _vertices.clear();
//   }
// 
//   void Expression(
//     std::shared_ptr<ranally::ExpressionVertex> const& vertex)
//   {
//     assert(vertex);
//     _vertices.push_back(vertex);
//   }
// 
//   std::vector<std::shared_ptr<ranally::ExpressionVertex>>
//     post_Targets()
//   {
//     assert(!_vertices.empty());
//     return _vertices;
//   }
// };


class If_pimpl:
    public ranally::If_pskel
{

public:

    void pre()
    {
        _data_stack.push(IfData());
    }

    void Expression(
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_data_stack.top().condition_vertex);
        assert(vertex);
        _data_stack.top().condition_vertex = vertex;
    }

    void Statements(
        std::vector<std::shared_ptr<ranally::StatementVertex>>
            const& vertices)
    {
        if(_data_stack.top().true_statement_vertices.empty()) {
            assert(!vertices.empty());
            _data_stack.top().true_statement_vertices = vertices;
        }
        else {
            assert(_data_stack.top().false_statement_vertices.empty());
            _data_stack.top().false_statement_vertices = vertices;
        }
    }

    std::shared_ptr<ranally::IfVertex> post_If()
    {
        assert(!_data_stack.empty());
        IfData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<ranally::IfVertex>(new ranally::IfVertex(
            result.condition_vertex, result.true_statement_vertices,
            result.false_statement_vertices));
    }

private:

    typedef std::vector<std::shared_ptr<ranally::StatementVertex>>
        StatementVertices;

    struct IfData
    {
        std::shared_ptr<ranally::ExpressionVertex> condition_vertex;
        StatementVertices true_statement_vertices;
        StatementVertices false_statement_vertices;
    };

    std::stack<IfData> _data_stack;

};


class While_pimpl:
    public ranally::While_pskel
{

public:

    void pre()
    {
        _data_stack.push(WhileData());
    }

    void Expression(
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_data_stack.top().condition_vertex);
        assert(vertex);
        _data_stack.top().condition_vertex = vertex;
    }

    void Statements(
        std::vector<std::shared_ptr<ranally::StatementVertex>>
            const& vertices)
    {
        if(_data_stack.top().true_statement_vertices.empty()) {
            assert(!vertices.empty());
            _data_stack.top().true_statement_vertices = vertices;
        }
        else {
            assert(_data_stack.top().false_statement_vertices.empty());
            _data_stack.top().false_statement_vertices = vertices;
        }
    }

    std::shared_ptr<ranally::WhileVertex> post_While()
    {
        assert(!_data_stack.empty());
        WhileData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<ranally::WhileVertex>(new ranally::WhileVertex(
            result.condition_vertex, result.true_statement_vertices,
                result.false_statement_vertices));
    }

private:

    typedef std::vector<std::shared_ptr<ranally::StatementVertex>>
        StatementVertices;

    struct WhileData
    {
        std::shared_ptr<ranally::ExpressionVertex> condition_vertex;
        StatementVertices true_statement_vertices;
        StatementVertices false_statement_vertices;
    };

    std::stack<WhileData> _data_stack;

};


class Statements_pimpl:
    public ranally::Statements_pskel
{

public:

    void pre()
    {
        _data_stack.push(StatementsData());
    }

    void Statement(
        std::shared_ptr<ranally::StatementVertex> const& vertex)
    {
        assert(vertex);
        _data_stack.top().push_back(vertex);
    }

    std::vector<std::shared_ptr<ranally::StatementVertex>>
        post_Statements()
    {
        assert(!_data_stack.empty());
        StatementsData result(_data_stack.top());
        _data_stack.pop();
        return result;
    }

private:

    typedef std::vector<std::shared_ptr<ranally::StatementVertex>>
        StatementsData;

    std::stack<StatementsData> _data_stack;

};


class Statement_pimpl:
    public ranally::Statement_pskel
{

public:

    void pre()
    {
        _data_stack.push(StatementData());
    }

    void line(
        unsigned long long line)
    {
        assert(!_data_stack.empty());
        _data_stack.top().line = line;
    }

    void col(
        unsigned long long col)
    {
        assert(!_data_stack.empty());
        _data_stack.top().col = col;
    }

    void Expression(
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void Assignment(
        std::shared_ptr<ranally::AssignmentVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void If(
        std::shared_ptr<ranally::IfVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void While(
        std::shared_ptr<ranally::WhileVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    std::shared_ptr<ranally::StatementVertex> post_Statement()
    {
        assert(!_data_stack.empty());
        StatementData result(_data_stack.top());
        _data_stack.pop();
        return result.vertex;
    }

private:

    struct StatementData
    {
        int line;
        int col;
        std::shared_ptr<ranally::StatementVertex> vertex;
    };

    std::stack<StatementData> _data_stack;

};


class Expressions_pimpl:
    public ranally::Expressions_pskel
{

public:

    void pre()
    {
        _data_stack.push(ExpressionsData());
    }

    void Expression(
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(vertex);
        _data_stack.top().push_back(vertex);
    }

    std::vector<std::shared_ptr<ranally::ExpressionVertex>>
        post_Expressions()
    {
        assert(!_data_stack.empty());
        ExpressionsData result(_data_stack.top());
        _data_stack.pop();
        return result;
    }

private:

    typedef std::vector<std::shared_ptr<ranally::ExpressionVertex>>
        ExpressionsData;

    std::stack<ExpressionsData> _data_stack;

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

    std::shared_ptr<ranally::ExpressionVertex> post_Integer()
    {
        std::shared_ptr<ranally::ExpressionVertex> result;

        switch(_size) {
            case 8: {
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<int8_t>(_value));
                break;
            }
            case 16: {
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<int16_t>(_value));
                break;
            }
            case 32: {
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<int32_t>(_value));
                break;
            }
            case 64: {
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<int64_t>(_value));
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

    std::shared_ptr<ranally::ExpressionVertex>
        post_UnsignedInteger()
    {
        std::shared_ptr<ranally::ExpressionVertex> result;

        switch(_size) {
            case 8: {
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<uint8_t>(_value));
                break;
            }
            case 16: {
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<uint16_t>(_value));
                break;
            }
            case 32: {
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<uint32_t>(_value));
                break;
            }
            case 64: {
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<uint64_t>(_value));
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

    std::shared_ptr<ranally::ExpressionVertex> post_Float()
    {
        std::shared_ptr<ranally::ExpressionVertex> result;

        switch(_size) {
            case 32: {
                assert(sizeof(float) == 4);
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<float>(_value));
                break;
            }
            case 64: {
                assert(sizeof(double) == 8);
                result = std::shared_ptr<ranally::ExpressionVertex>(
                    new ranally::NumberVertex<double>(_value));
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
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_vertex);
        _vertex = vertex;
    }

    void UnsignedInteger(
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_vertex);
        _vertex = vertex;
    }

    void Float(
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_vertex);
        _vertex = vertex;
    }

    std::shared_ptr<ranally::ExpressionVertex> post_Number()
    {
        assert(_vertex);
        return _vertex;
    }

private:

    std::shared_ptr<ranally::ExpressionVertex> _vertex;

};


class Function_pimpl:
    public ranally::Function_pskel
{

public:

    void pre()
    {
        _data_stack.push(FunctionData());
    }

    void Name(
        std::string const& name)
    {
        assert(!_data_stack.empty());
        _data_stack.top().name = ranally::String(name);
    }

    void Expressions(
        std::vector<std::shared_ptr<ranally::ExpressionVertex>>
            const& vertices)
    {
        assert(!_data_stack.empty());
        assert(_data_stack.top().expression_vertices.empty());
        _data_stack.top().expression_vertices = vertices;
    }

    std::shared_ptr<ranally::FunctionVertex> post_Function()
    {
        assert(!_data_stack.empty());
        FunctionData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<ranally::FunctionVertex>(
            new ranally::FunctionVertex(result.name,
                result.expression_vertices));
    }

private:

    typedef std::vector<std::shared_ptr<ranally::ExpressionVertex>>
        ExpressionVertices;

    struct FunctionData
    {
        ranally::String name;
        ExpressionVertices expression_vertices;
    };

    std::stack<FunctionData> _data_stack;

};


class Operator_pimpl:
    public ranally::Operator_pskel
{

public:

    void pre()
    {
        _data_stack.push(OperatorData());
    }

    void Name(
        std::string const& name)
    {
        assert(!_data_stack.empty());
        _data_stack.top().name = ranally::String(name);
    }

    void Expressions(
        std::vector<std::shared_ptr<ranally::ExpressionVertex>>
            const& vertices)
    {
        assert(!_data_stack.empty());
        assert(_data_stack.top().expression_vertices.empty());
        _data_stack.top().expression_vertices = vertices;
    }

    std::shared_ptr<ranally::OperatorVertex> post_Operator()
    {
        assert(!_data_stack.empty());
        OperatorData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<ranally::OperatorVertex>(
            new ranally::OperatorVertex(result.name,
                result.expression_vertices));
    }

private:

    typedef std::vector<std::shared_ptr<ranally::ExpressionVertex>>
        ExpressionVertices;

    struct OperatorData
    {
        ranally::String name;
        ExpressionVertices expression_vertices;
    };

    std::stack<OperatorData> _data_stack;

};


class Subscript_pimpl:
    public ranally::Subscript_pskel
{

public:

    void pre()
    {
        _data_stack.push(SubscriptData());
    }

    void Expression(
        std::shared_ptr<ranally::ExpressionVertex>
            const& vertex)
    {
        assert(_data_stack.size() == 1);
        if(!_data_stack.top().expression) {
            _data_stack.top().expression = vertex;
        }
        else {
            assert(!_data_stack.top().selection);
            _data_stack.top().selection = vertex;
        }
    }

    std::shared_ptr<ranally::ExpressionVertex> post_Subscript()
    {
        assert(!_data_stack.empty());
        SubscriptData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<ranally::SubscriptVertex>(
            new ranally::SubscriptVertex(result.expression, result.selection));
    }

private:

    struct SubscriptData
    {
        ranally::ExpressionVertexPtr expression;
        ranally::ExpressionVertexPtr selection;
    };

    std::stack<SubscriptData> _data_stack;

};


class Expression_pimpl:
    public ranally::Expression_pskel
{

public:

    void pre()
    {
        _data_stack.push(ExpressionData());
    }

    void line(
        unsigned long long line)
    {
        assert(!_data_stack.empty());
        _data_stack.top().line = line;
    }

    void col(
        unsigned long long col)
    {
        assert(!_data_stack.empty());
        _data_stack.top().col = col;
    }

    void Name(
        std::string const& name)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = std::shared_ptr<ranally::NameVertex>(
            new ranally::NameVertex(_data_stack.top().line,
                _data_stack.top().col, ranally::String(name)));
    }

    void Subscript(
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void String(
        std::string const& string)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = std::shared_ptr<ranally::StringVertex>(
            new ranally::StringVertex(_data_stack.top().line,
                _data_stack.top().col, ranally::String(string)));
    }

    void Number(
        std::shared_ptr<ranally::ExpressionVertex> const& vertex)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void Function(
        std::shared_ptr<ranally::FunctionVertex> const& vertex)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void Operator(
        std::shared_ptr<ranally::OperatorVertex> const& vertex)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    std::shared_ptr<ranally::ExpressionVertex> post_Expression()
    {
        assert(!_data_stack.empty());
        ExpressionData result(_data_stack.top());
        _data_stack.pop();
        return result.vertex;
    }

private:

  struct ExpressionData
  {
      int line;
      int col;
      std::shared_ptr<ranally::ExpressionVertex> vertex;
  };

  std::stack<ExpressionData> _data_stack;

};

} // Anonymous namespace


namespace ranally {

//! Parse the Xml in \a stream and return a syntax tree.
/*!
  \param     stream Stream with Xml to parse.
  \return    Root of syntax tree.
  \exception std::exception In case of a System category error.
  \exception xml_schema::parsing In case of Xml category error.
*/
std::shared_ptr<ScriptVertex> XmlParser::parse(
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

    Subscript_pimpl subscript_p;
    subscript_p.parsers(expression_p);

    expression_p.parsers(string_p, subscript_p, string_p, number_p,
        function_p, operator_p, non_negative_integer_p,
        non_negative_integer_p);

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
    statement_p.parsers(expression_p, assignment_p, if_p, while_p,
        non_negative_integer_p, non_negative_integer_p);

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
std::shared_ptr<ScriptVertex> XmlParser::parse_string(
    String const& xml) const
{
    // Copy string contents in a string stream and work with that.
    std::stringstream stream;
    stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    stream << xml.encode_in_utf8(); // << std::endl;

    std::shared_ptr<ScriptVertex> vertex;

    try {
        vertex = parse(stream);
    }
    catch(xml_schema::parsing const& exception) {
        assert(!exception.diagnostics().empty());
        BOOST_THROW_EXCEPTION(ranally::detail::ParseError()
            << detail::ExceptionSourceName("<string>")
            << detail::ExceptionLineNr(exception.diagnostics()[0].line())
            << detail::ExceptionColNr(exception.diagnostics()[0].column())
            << detail::ExceptionMessage(exception.diagnostics()[0].message())
        );
    }

    return vertex;
}

} // namespace ranally
