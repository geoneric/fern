#pragma once
#include "fern/expression_tree/arity.h"
#include "fern/expression_tree/array.h"
#include "fern/expression_tree/constant.h"


namespace fern {

template<
    class U,
    class V>
struct Times
{
};


template<
    class U,
    class V>
struct Times<Constant<U>, Constant<V>>
{
    typedef Constant<U> A1;
    typedef Constant<V> A2;
    typedef Constant<decltype(U() * V())> result_type;

    typedef arity::Binary Arity;

    result_type operator()(
        Constant<U> const& argument1,
        Constant<V> const& argument2) const
    {
        return argument1.value * argument2.value;
    }
};


// template<
//     class U,
//     class V>
// struct Times<Array<U>, Array<V>>
// {
//     typedef Array<typename Times<U, V>::result_type> result_type;
// };

} // namespace fern
