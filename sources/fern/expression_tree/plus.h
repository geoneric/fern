#pragma once
#include "fern/expression_tree/arity.h"
#include "fern/expression_tree/array.h"
#include "fern/expression_tree/constant.h"


namespace fern {

template<
    class U,
    class V>
struct Plus
{

    typedef U A1;
    typedef V A2;
    typedef decltype(U() + V()) result_type;
    typedef arity::Binary Arity;

};


template<
    class U,
    class V>
struct Plus<Constant<U>, Constant<V>>
{

    typedef Constant<U> A1;
    typedef Constant<V> A2;
    typedef Constant<typename Plus<U, V>::result_type> result_type;
    typedef typename Plus<U, V>::Arity Arity;

    result_type operator()(
        Constant<U> const& argument1,
        Constant<V> const& argument2) const
    {
        return argument1.value + argument2.value;
    }

};


template<
    class U,
    class V>
struct Plus<Array<U>, Array<V>>
{

    typedef Array<U> A1;
    typedef Array<V> A2;
    typedef Array<typename Plus<U, V>::result_type> result_type;
    typedef typename Plus<U, V>::Arity Arity;

    result_type operator()(
        Array<U> const& argument1,
        Array<V> const& argument2) const
    {
        // TODO
        // Array<U>::const_iterator it1 = argument1.begin();
        // Array<U>::const_iterator end1 = argument1.end();
        // Array<V>::const_iterator it2 = argument2.begin();

        // result_type result;

        // for(; it1 != end1; ++it1, ++it2) {
        // }

        // for(Index i = 0; i != nr_
        // return argument1.container + argument2.container;
    }

};

} // namespace fern
