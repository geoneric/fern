#pragma once
#include "fern/expression_tree/arity.h"
#include "fern/expression_tree/array.h"
#include "fern/expression_tree/constant.h"


namespace fern {

template<
    class T>
struct Sqrt
{
};


template<
    class T>
struct Sqrt<Constant<T>>
{
    typedef Constant<T> A;
    typedef Constant<double> result_type;

    typedef arity::Unary Arity;

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
//     typedef Array<typename Sqrt<T>::result_type> result_type;
// };

} // namespace fern
