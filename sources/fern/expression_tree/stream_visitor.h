#pragma once
#include <boost/variant/static_visitor.hpp>
#include "fern/expression_tree/ast.h"


namespace fern {
namespace expression_tree {

#define STREAM_CONSTANT(                                                       \
    Type)                                                                      \
void stream(                                                                   \
    Type const& value,                                                         \
    std::ostream& stream)                                                      \
{                                                                              \
    stream                                                                     \
        << #Type"("                                                            \
        << value                                                               \
        << ')'                                                                 \
        ;                                                                      \
}

STREAM_CONSTANT(int8_t)
STREAM_CONSTANT(int16_t)
STREAM_CONSTANT(int32_t)
STREAM_CONSTANT(int64_t)
STREAM_CONSTANT(uint8_t)
STREAM_CONSTANT(uint16_t)
STREAM_CONSTANT(uint32_t)
STREAM_CONSTANT(uint64_t)
STREAM_CONSTANT(float)
STREAM_CONSTANT(double)

#undef STREAM_CONSTANT


// template<
//     class T>
// void stream(
//     Array<T> const& /* array */,
//     std::ostream& stream)
// {
//     stream << "array";
// }


class StreamVisitor:
    public boost::static_visitor<void>
{

public:

    StreamVisitor(
        std::ostream& stream)

        : _stream(stream)

    {
    }

    template<
        class T>
    void operator()(
        Constant<T> const& constant) const
    {
        stream(constant.value, _stream);
    }

    void operator()(
        Data const& data) const
    {
        // Argument is a variant, visit it.
        boost::apply_visitor(*this, data);
    }

    template<
        class T>
    void operator()(
        Operation<T> const& operation) const
    {
        // stream<T>(_stream);

        _stream << operation.name << "(";

        if(!operation.expressions.empty()) {
            // Expression is a variant, visit it.
            boost::apply_visitor(*this, operation.expressions[0]);
        }

        for(size_t i = 1; i < operation.expressions.size(); ++i) {
            _stream << ", ";
            boost::apply_visitor(*this, operation.expressions[i]);
        }

        _stream << ")";
    }

private:

    std::ostream&  _stream;

};


void stream(
    Expression const& expression,
    std::ostream& stream)
{
    stream
        << '('
        ;

    boost::apply_visitor(StreamVisitor(stream), expression);

    stream
        << ')'
        ;
}

} // namespace expression_tree
} // namespace fern
