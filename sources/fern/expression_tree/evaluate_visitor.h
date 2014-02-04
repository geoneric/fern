#pragma once
#include <boost/variant/static_visitor.hpp>
#include "fern/expression_tree/ast.h"


namespace fern {

class EvaluateVisitor:
    public boost::static_visitor<Data>
{

public:

    EvaluateVisitor()
    {
    }

    // template<
    //     class T>
    // Data operator()(
    //     Constant<T> const& constant) const
    // {
    //     return constant;
    // }

    // template<
    //     class T>
    // Data operator()(
    //     Array<T> const& array) const
    // {
    //     return array;
    // }

    Data operator()(
        Data const& data) const
    {
        // // Argument is a variant, visit it.
        // return boost::apply_visitor(*this, data);
        return data;
    }

    template<
        class T>
    Data operator()(
        Operation<T> const& operation) const
    {
        std::vector<Data> data;

        for(auto& expression: operation.expressions) {
            data.emplace_back(boost::apply_visitor(*this, expression));
        }

        return operation.implementation.evaluate(data);
    }

private:

};


Data evaluate(
    Expression const& expression)
{
    return boost::apply_visitor(EvaluateVisitor(), expression);
}

} // namespace fern
