#pragma once
#include "fern/expression_tree/arity.h"
#include "fern/expression_tree/array.h"
#include "fern/expression_tree/constant.h"


namespace fern {
namespace expression_tree {

template<
    class T>
struct Sqrt
{
};


template<
    class T>
struct Sqrt<Constant<T>>
{
    using A = Constant<T>;
    using result_type = Constant<double>;

    using Category = Local;
    using Arity = arity::Unary;

    result_type operator()(
        Constant<T> const& argument) const
    {
        return std::sqrt(argument.value);
    }
};


// template<
//     class T>
// struct Sqrt<Array<T>>
// {
//     using result_type = Array<typename Sqrt<T>::result_type>;
// };

} // namespace expression_tree
} // namespace fern
