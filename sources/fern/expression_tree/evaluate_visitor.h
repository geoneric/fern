#pragma once
#include <boost/variant/static_visitor.hpp>
#include "fern/expression_tree/ast.h"


namespace fern {
namespace expression_tree {

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
        // Figure out how this should work. With threading and all.

        // - Operations do their thing, for the whole raster, or for a piece.
        // - They can be visited one after the other, depth first.
        // - After visiting an operation, the result raster is available as an
        //   argument for parent operations.

        // - There are multiple threading strategies:
        //   - block:
        //     - local operations
        //     - aggregating operations
        //   - areas:
        //     - neighborhood operations
        // - Tasks can be configured that do stuff for a specific area. When
        //   evaluating an operation that cannot be split like surrounding
        //   operations, a synchronization task must be inserted to the task
        //   list.
        // - At first, split per operation?!

        // - Perform a depth-first visit. Each operation can start using
        //   threads, but they have to wait untill all arguments are evaluated.
        // - This creates a lot of thread, but that's OK for now.

        // - Split off into operation categories.

        // T is a Constant<V> or Array<V>.
        std::vector<Data> data;

        for(auto& expression: operation.expressions) {
            data.emplace_back(boost::apply_visitor(*this, expression));
        }

        // TODO When evaluating, we need to pass in the location of the result
        //      too.
        // TODO Use threads to split up the work. Pass the extent of the block
        //      to evaluate.
        return operation.implementation.evaluate(data);
    }

private:

};


Data evaluate(
    Expression const& expression)
{
    return boost::apply_visitor(EvaluateVisitor(), expression);
}

} // namespace expression_tree
} // namespace fern
