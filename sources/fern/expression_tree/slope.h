#pragma once
#include "fern/expression_tree/arity.h"
#include "fern/expression_tree/array.h"
#include "fern/expression_tree/constant.h"


namespace fern {
namespace expression_tree {

template<
    class T>
struct Slope
{
};


template<
    class T>
struct Slope<Constant<T>>
{

    using A = Constant<T>;
    using result_type = Constant<double>;
    using Arity = arity::Unary;

    result_type operator()(
        Constant<T> const& /* argument */) const
    {
        return result_type(0);
    }

};


// template<
//     class T>
// struct Slope<Array<T>>
// {
// 
//     using A = Array<T>;
//     using result_type = Array<double>;
//     using Arity = arity::Unary;
// 
//     result_type operator()(
//         Array<T> const& /* argument */) const
//     {
//         assert(false);
//         return result_type();
//     }
// 
// };

} // namespace expression_tree
} // namespace fern
