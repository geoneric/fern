#pragma once
#include "fern/expression_tree/arity.h"
#include "fern/expression_tree/array.h"
#include "fern/expression_tree/constant.h"


namespace fern {

template<
    class T>
struct Slope
{
};


template<
    class T>
struct Slope<Constant<T>>
{

    typedef Constant<T> A;
    typedef Constant<double> result_type;
    typedef arity::Unary Arity;

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
//     typedef Array<T> A;
//     typedef Array<double> result_type;
//     typedef arity::Unary Arity;
// 
//     result_type operator()(
//         Array<T> const& /* argument */) const
//     {
//         assert(false);
//         return result_type();
//     }
// 
// };

} // namespace fern
