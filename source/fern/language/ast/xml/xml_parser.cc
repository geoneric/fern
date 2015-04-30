// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/xml/xml_parser.h"
#include <sstream>
#include <stack>
#include "fern/core/exception.h"
#include "fern/language/ast/core/vertices.h"
#include "fern/language/ast/xml/syntax_tree-pskel.hxx"


namespace {

class Fern_pimpl:
    public fern::Fern_pskel
{

public:

    void pre()
    {
        _scope_vertex.reset();
    }

    void source(
        std::string const& sourceName)
    {
        _source_name = fern::String(sourceName);
    }

    void Statements(
        std::vector<std::shared_ptr<fern::StatementVertex>> const&
            statements)
    {
        assert(!_scope_vertex);
        _scope_vertex = std::make_shared<fern::ScopeVertex>(statements);
    }

    std::shared_ptr<fern::ModuleVertex> post_Fern()
    {
        return std::shared_ptr<fern::ModuleVertex>(
            std::make_shared<fern::ModuleVertex>(_source_name, _scope_vertex));
    }

private:

    fern::String  _source_name;

    std::shared_ptr<fern::ScopeVertex> _scope_vertex;

};


class Assignment_pimpl:
    public fern::Assignment_pskel
{

public:

    void pre()
    {
        _expressions.clear();
    }

    // void Targets(
    //   std::vector<std::shared_ptr<fern::ExpressionVertex>> const&
    //     vertices)
    // {
    //   assert(!vertices.empty());
    //   _targets = vertices;
    // }

    void Expression(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        _expressions.emplace_back(vertex);
        assert(_expressions.size() <= 2);
    }

    std::shared_ptr<fern::AssignmentVertex> post_Assignment()
    {
        assert(!_expressions.empty());
        return std::shared_ptr<fern::AssignmentVertex>(
            std::make_shared<fern::AssignmentVertex>(_expressions[0],
                _expressions[1]));
    }

private:

    std::vector<std::shared_ptr<fern::ExpressionVertex>>
        _expressions;

};


// class Targets_pimpl: public fern::Targets_pskel
// {
// private:
//   std::vector<std::shared_ptr<fern::ExpressionVertex>>
//     _vertices;
// 
// public:
//   void pre()
//   {
//     _vertices.clear();
//   }
// 
//   void Expression(
//     std::shared_ptr<fern::ExpressionVertex> const& vertex)
//   {
//     assert(vertex);
//     _vertices.emplace_back(vertex);
//   }
// 
//   std::vector<std::shared_ptr<fern::ExpressionVertex>>
//     post_Targets()
//   {
//     assert(!_vertices.empty());
//     return _vertices;
//   }
// };


class If_pimpl:
    public fern::If_pskel
{

public:

    void pre()
    {
        _data_stack.push(IfData());
    }

    void Expression(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(!_data_stack.top().condition_vertex);
        assert(vertex);
        _data_stack.top().condition_vertex = vertex;
    }

    void Statements(
        std::vector<std::shared_ptr<fern::StatementVertex>>
            const& statements)
    {
        if(!_data_stack.top().true_scope) {
            assert(!statements.empty());
            assert(!_data_stack.top().false_scope);
            _data_stack.top().true_scope = std::make_shared<fern::ScopeVertex>(
                statements);
        }
        else {
            assert(!_data_stack.top().false_scope);
            _data_stack.top().false_scope = std::make_shared<fern::ScopeVertex>(
                statements);
        }
    }

    std::shared_ptr<fern::IfVertex> post_If()
    {
        assert(!_data_stack.empty());
        IfData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<fern::IfVertex>(std::make_shared<fern::IfVertex>(
            result.condition_vertex, result.true_scope, result.false_scope));
    }

private:

    struct IfData
    {
        std::shared_ptr<fern::ExpressionVertex> condition_vertex;
        std::shared_ptr<fern::ScopeVertex> true_scope;
        std::shared_ptr<fern::ScopeVertex> false_scope;
    };

    std::stack<IfData> _data_stack;

};


class While_pimpl:
    public fern::While_pskel
{

public:

    void pre()
    {
        _data_stack.push(WhileData());
    }

    void Expression(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(!_data_stack.top().condition_vertex);
        assert(vertex);
        _data_stack.top().condition_vertex = vertex;
    }

    void Statements(
        std::vector<std::shared_ptr<fern::StatementVertex>>
            const& statements)
    {
        if(!_data_stack.top().true_scope) {
            assert(!statements.empty());
            assert(!_data_stack.top().false_scope);
            _data_stack.top().true_scope = std::make_shared<fern::ScopeVertex>(
                statements);
        }
        else {
            assert(!_data_stack.top().false_scope);
            _data_stack.top().false_scope = std::make_shared<fern::ScopeVertex>(
                statements);
        }
    }

    std::shared_ptr<fern::WhileVertex> post_While()
    {
        assert(!_data_stack.empty());
        WhileData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<fern::WhileVertex>(
            std::make_shared<fern::WhileVertex>(result.condition_vertex,
                result.true_scope, result.false_scope));
    }

private:

    struct WhileData
    {
        std::shared_ptr<fern::ExpressionVertex> condition_vertex;
        std::shared_ptr<fern::ScopeVertex> true_scope;
        std::shared_ptr<fern::ScopeVertex> false_scope;
    };

    std::stack<WhileData> _data_stack;

};


class FunctionDefinition_pimpl:
    public fern::FunctionDefinition_pskel
{

public:

    void pre()
    {
        _data_stack.push(FunctionDefinitionData());
    }

    void Name(
        std::string const& name)
    {
        assert(!_data_stack.empty());
        _data_stack.top().name = fern::String(name);
    }

    void Expressions(
        std::vector<std::shared_ptr<fern::ExpressionVertex>>
            const& vertices)
    {
        assert(!_data_stack.empty());
        assert(_data_stack.top().expression_vertices.empty());
        _data_stack.top().expression_vertices = vertices;
    }

    void Statements(
        std::vector<std::shared_ptr<fern::StatementVertex>>
            const& statements)
    {
        assert(!statements.empty());
        _data_stack.top().scope_vertex = std::make_shared<fern::ScopeVertex>(
            statements);
    }

    std::shared_ptr<fern::FunctionDefinitionVertex> post_FunctionDefinition()
    {
        assert(!_data_stack.empty());
        FunctionDefinitionData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<fern::FunctionDefinitionVertex>(
            std::make_shared<fern::FunctionDefinitionVertex>(
                result.name, result.expression_vertices, result.scope_vertex));
    }

private:

    using ExpressionVertices = std::vector<std::shared_ptr<fern::ExpressionVertex>>;

    struct FunctionDefinitionData
    {
        fern::String name;
        ExpressionVertices expression_vertices;
        std::shared_ptr<fern::ScopeVertex> scope_vertex;
    };

    std::stack<FunctionDefinitionData> _data_stack;

};


class Return_pimpl:
    public fern::Return_pskel
{

public:

    void pre()
    {
        _data_stack.push(ReturnData());
    }

    void Expression(
        std::shared_ptr<fern::ExpressionVertex> const& expression)
    {
        assert(!_data_stack.empty());
        _data_stack.top().expression_vertex = expression;
    }

    std::shared_ptr<fern::ReturnVertex> post_Return()
    {
        assert(!_data_stack.empty());
        ReturnData result(_data_stack.top());
        _data_stack.pop();
        std::shared_ptr<fern::ReturnVertex> vertex(
            result.expression_vertex
                ?  std::make_shared<fern::ReturnVertex>(
                       result.expression_vertex)
                :  std::make_shared<fern::ReturnVertex>());
        return vertex;
    }

private:

    struct ReturnData
    {
        std::shared_ptr<fern::ExpressionVertex> expression_vertex;
    };

    std::stack<ReturnData> _data_stack;

};


class Statements_pimpl:
    public fern::Statements_pskel
{

public:

    void pre()
    {
        _data_stack.push(StatementsData());
    }

    void Statement(
        std::shared_ptr<fern::StatementVertex> const& vertex)
    {
        assert(vertex);
        _data_stack.top().emplace_back(vertex);
    }

    std::vector<std::shared_ptr<fern::StatementVertex>>
        post_Statements()
    {
        assert(!_data_stack.empty());
        StatementsData result(_data_stack.top());
        _data_stack.pop();
        return result;
    }

private:

    using StatementsData = std::vector<std::shared_ptr<fern::StatementVertex>>;

    std::stack<StatementsData> _data_stack;

};


class Statement_pimpl:
    public fern::Statement_pskel
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
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void Assignment(
        std::shared_ptr<fern::AssignmentVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void If(
        std::shared_ptr<fern::IfVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void While(
        std::shared_ptr<fern::WhileVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void FunctionDefinition(
        std::shared_ptr<fern::FunctionDefinitionVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void Return(
        std::shared_ptr<fern::ReturnVertex> const& vertex)
    {
        assert(vertex);
        assert(!_data_stack.empty());
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    std::shared_ptr<fern::StatementVertex> post_Statement()
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
        std::shared_ptr<fern::StatementVertex> vertex;
    };

    std::stack<StatementData> _data_stack;

};


class Expressions_pimpl:
    public fern::Expressions_pskel
{

public:

    void pre()
    {
        _data_stack.push(ExpressionsData());
    }

    void Expression(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(vertex);
        _data_stack.top().emplace_back(vertex);
    }

    std::vector<std::shared_ptr<fern::ExpressionVertex>>
        post_Expressions()
    {
        assert(!_data_stack.empty());
        ExpressionsData result(_data_stack.top());
        _data_stack.pop();
        return result;
    }

private:

    using ExpressionsData = std::vector<std::shared_ptr<
        fern::ExpressionVertex>>;

    std::stack<ExpressionsData> _data_stack;

};


class Integer_pimpl:
    public fern::Integer_pskel
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

    std::shared_ptr<fern::ExpressionVertex> post_Integer()
    {
        std::shared_ptr<fern::ExpressionVertex> result;

        switch(_size) {
            case 8: {
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<int8_t>>(_value));
                break;
            }
            case 16: {
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<int16_t>>(_value));
                break;
            }
            case 32: {
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<int32_t>>(_value));
                break;
            }
            case 64: {
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<int64_t>>(_value));
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
    public fern::UnsignedInteger_pskel
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

    std::shared_ptr<fern::ExpressionVertex>
        post_UnsignedInteger()
    {
        std::shared_ptr<fern::ExpressionVertex> result;

        switch(_size) {
            case 8: {
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<uint8_t>>(_value));
                break;
            }
            case 16: {
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<uint16_t>>(_value));
                break;
            }
            case 32: {
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<uint32_t>>(_value));
                break;
            }
            case 64: {
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<uint64_t>>(_value));
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
    public fern::Float_pskel
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

    std::shared_ptr<fern::ExpressionVertex> post_Float()
    {
        std::shared_ptr<fern::ExpressionVertex> result;

        switch(_size) {
            case 32: {
                assert(sizeof(float) == 4);
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<float>>(_value));
                break;
            }
            case 64: {
                assert(sizeof(double) == 8);
                result = std::shared_ptr<fern::ExpressionVertex>(
                    std::make_shared<fern::NumberVertex<double>>(_value));
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
    public fern::Number_pskel
{

public:

    void pre()
    {
        _vertex.reset();
    }

    void Integer(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(!_vertex);
        _vertex = vertex;
    }

    void UnsignedInteger(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(!_vertex);
        _vertex = vertex;
    }

    void Float(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(!_vertex);
        _vertex = vertex;
    }

    std::shared_ptr<fern::ExpressionVertex> post_Number()
    {
        assert(_vertex);
        return _vertex;
    }

private:

    std::shared_ptr<fern::ExpressionVertex> _vertex;

};


class FunctionCall_pimpl:
    public fern::FunctionCall_pskel
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
        _data_stack.top().name = fern::String(name);
    }

    void Expressions(
        std::vector<std::shared_ptr<fern::ExpressionVertex>>
            const& vertices)
    {
        assert(!_data_stack.empty());
        assert(_data_stack.top().expression_vertices.empty());
        _data_stack.top().expression_vertices = vertices;
    }

    std::shared_ptr<fern::FunctionCallVertex> post_FunctionCall()
    {
        assert(!_data_stack.empty());
        FunctionData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<fern::FunctionCallVertex>(
            std::make_shared<fern::FunctionCallVertex>(result.name,
                result.expression_vertices));
    }

private:

    using ExpressionVertices = std::vector<std::shared_ptr<
        fern::ExpressionVertex>>;

    struct FunctionData
    {
        fern::String name;
        ExpressionVertices expression_vertices;
    };

    std::stack<FunctionData> _data_stack;

};


class Operator_pimpl:
    public fern::Operator_pskel
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
        _data_stack.top().name = fern::String(name);
    }

    void Expressions(
        std::vector<std::shared_ptr<fern::ExpressionVertex>>
            const& vertices)
    {
        assert(!_data_stack.empty());
        assert(_data_stack.top().expression_vertices.empty());
        _data_stack.top().expression_vertices = vertices;
    }

    std::shared_ptr<fern::OperatorVertex> post_Operator()
    {
        assert(!_data_stack.empty());
        OperatorData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<fern::OperatorVertex>(
            std::make_shared<fern::OperatorVertex>(result.name,
                result.expression_vertices));
    }

private:

    using ExpressionVertices = std::vector<std::shared_ptr<
        fern::ExpressionVertex>>;

    struct OperatorData
    {
        fern::String name;
        ExpressionVertices expression_vertices;
    };

    std::stack<OperatorData> _data_stack;

};


class Subscript_pimpl:
    public fern::Subscript_pskel
{

public:

    void pre()
    {
        _data_stack.push(SubscriptData());
    }

    void Expression(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
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

    std::shared_ptr<fern::ExpressionVertex> post_Subscript()
    {
        assert(!_data_stack.empty());
        SubscriptData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<fern::SubscriptVertex>(
            std::make_shared<fern::SubscriptVertex>(result.expression,
                result.selection));
    }

private:

    struct SubscriptData
    {
        fern::ExpressionVertexPtr expression;
        fern::ExpressionVertexPtr selection;
    };

    std::stack<SubscriptData> _data_stack;

};


class Attribute_pimpl:
    public fern::Attribute_pskel
{

public:

    void pre()
    {
        _data_stack.push(AttributeData());
    }

    void Expression(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(_data_stack.size() == 1);
        assert(!_data_stack.top().expression);
        _data_stack.top().expression = vertex;
    }

    void Name(
        std::string const& name)
    {
        assert(_data_stack.size() == 1);
        assert(_data_stack.top().member_name.empty());
        _data_stack.top().member_name = name;
    }

    std::shared_ptr<fern::ExpressionVertex> post_Attribute()
    {
        assert(_data_stack.size() == 1);
        AttributeData result(_data_stack.top());
        _data_stack.pop();
        return std::shared_ptr<fern::AttributeVertex>(
            std::make_shared<fern::AttributeVertex>(result.expression,
                result.member_name));
    }

private:

    struct AttributeData
    {
        fern::ExpressionVertexPtr expression;
        std::string member_name;
    };

    std::stack<AttributeData> _data_stack;

};


class Expression_pimpl:
    public fern::Expression_pskel
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
        _data_stack.top().vertex = std::shared_ptr<fern::NameVertex>(
            std::make_shared<fern::NameVertex>(_data_stack.top().line,
                _data_stack.top().col, fern::String(name)));
    }

    void Subscript(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void Attribute(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
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
        _data_stack.top().vertex = std::shared_ptr<fern::StringVertex>(
            std::make_shared<fern::StringVertex>(_data_stack.top().line,
                _data_stack.top().col, fern::String(string)));
    }

    void Number(
        std::shared_ptr<fern::ExpressionVertex> const& vertex)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void FunctionCall(
        std::shared_ptr<fern::FunctionCallVertex> const& vertex)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    void Operator(
        std::shared_ptr<fern::OperatorVertex> const& vertex)
    {
        assert(!_data_stack.empty());
        assert(!_data_stack.top().vertex);
        _data_stack.top().vertex = vertex;
        _data_stack.top().vertex->set_position(_data_stack.top().line,
            _data_stack.top().col);
    }

    std::shared_ptr<fern::ExpressionVertex> post_Expression()
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
      std::shared_ptr<fern::ExpressionVertex> vertex;
  };

  std::stack<ExpressionData> _data_stack;

};

} // Anonymous namespace


namespace fern {

//! Parse the Xml in \a stream and return a syntax tree.
/*!
  \param     stream Stream with Xml to parse.
  \return    Root of syntax tree.
  \exception std::exception In case of a System category error.
  \exception xml_schema::parsing In case of Xml category error.
*/
std::shared_ptr<ModuleVertex> XmlParser::parse(
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

    FunctionCall_pimpl function_p;
    function_p.parsers(string_p, expressions_p);

    Operator_pimpl operator_p;
    operator_p.parsers(string_p, expressions_p);

    Subscript_pimpl subscript_p;
    subscript_p.parsers(expression_p);

    Attribute_pimpl attribute_p;
    attribute_p.parsers(expression_p, string_p);

    expression_p.parsers(string_p, subscript_p, attribute_p, string_p, number_p,
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

    FunctionDefinition_pimpl function_definition_p;
    function_definition_p.parsers(string_p, expressions_p, statements_p);

    Return_pimpl return_p;
    return_p.parsers(expression_p);

    Statement_pimpl statement_p;
    statement_p.parsers(expression_p, assignment_p, if_p, while_p,
        function_definition_p, return_p, non_negative_integer_p,
        non_negative_integer_p);

    statements_p.parsers(statement_p);

    Fern_pimpl fern_p;
    fern_p.parsers(statements_p, string_p);

    xml_schema::document doc_p(fern_p, "Fern");

    fern_p.pre();
    doc_p.parse(stream);
    return fern_p.post_Fern();
}


/*!
  \overload
  \param     xml String with Xml to parse.
*/
std::shared_ptr<ModuleVertex> XmlParser::parse_string(
    String const& xml) const
{
    // Copy string contents in a string stream and work with that.
    std::stringstream stream;
    stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    stream << xml.encode_in_utf8(); // << std::endl;

    std::shared_ptr<ModuleVertex> vertex;

    try {
        vertex = parse(stream);
    }
    catch(xml_schema::parsing const& exception) {
        assert(!exception.diagnostics().empty());
        BOOST_THROW_EXCEPTION(fern::detail::ParseError()
            << detail::ExceptionSourceName("<string>")
            << detail::ExceptionLineNr(exception.diagnostics()[0].line())
            << detail::ExceptionColNr(exception.diagnostics()[0].column())
            << detail::ExceptionMessage(exception.diagnostics()[0].message())
        );
    }

    return vertex;
}

} // namespace fern
