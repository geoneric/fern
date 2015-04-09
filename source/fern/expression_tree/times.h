// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/expression_tree/arity.h"
#include "fern/expression_tree/array.h"
#include "fern/expression_tree/constant.h"


namespace fern {
namespace expression_tree {

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
    using A1 = Constant<U>;
    using A2 = Constant<V>;
    using result_type = Constant<decltype(U() * V())>;

    using Category = Local;
    using Arity = arity::Binary;

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
//     using result_type = Array<typename Times<U, V>::result_type>;
// };

} // namespace expression_tree
} // namespace fern
