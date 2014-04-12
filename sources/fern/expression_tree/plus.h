#pragma once
#include "fern/expression_tree/arity.h"
#include "fern/expression_tree/ast.h"
// #include "fern/expression_tree/category.h"
#include "fern/expression_tree/evaluate_visitor.h"


namespace fern {
namespace expression_tree {

template<
    class U,
    class V>
struct Plus
{

    // TODO
    // Use algorithm from algorithm library.

    static_assert(std::is_arithmetic<U>::value, "Type must be numeric");
    static_assert(std::is_arithmetic<V>::value, "Type must be numeric");

    using A1 = U;
    using A2 = V;
    using result_type = decltype(U() + V());
    // using Category = Local;
    using Arity = arity::Binary;

};


template<
    class U,
    class V>
struct Plus<Constant<U>, Constant<V>>
{

    using A1 = Constant<U>;
    using A2 = Constant<V>;
    using result_type = Constant<typename Plus<U, V>::result_type>;
    // using Category = typename Plus<U, V>::Category;
    using Arity = typename Plus<U, V>::Arity;

    result_type operator()(
        Constant<U> const& argument1,
        Constant<V> const& argument2) const
    {
        // TODO
        // Use algorithm from algorithm library.

        return argument1.value + argument2.value;
    }

};


// template<
//     class U,
//     class V>
// struct Plus<Array<U>, Array<V>>
// {
// 
//     using A1 = Array<U>;
//     using A2 = Array<V>;
//     using result_type = Array<typename Plus<U, V>::result_type>;
//     using Arity = typename Plus<U, V>::Arity;
// 
//     result_type operator()(
//         Array<U> const& /* argument1 */,
//         Array<V> const& /* argument2 */) const
//     {
//         // TODO
//         // Array<U>::const_iterator it1 = argument1.begin();
//         // Array<U>::const_iterator end1 = argument1.end();
//         // Array<V>::const_iterator it2 = argument2.begin();
// 
//         // result_type result;
// 
//         // for(; it1 != end1; ++it1, ++it2) {
//         // }
// 
//         // for(Index i = 0; i != nr_
//         // return argument1.container + argument2.container;
// 
//         return result_type(std::vector<typename result_type::value_type>());
//     }
// 
// };


template<
    class U,
    class V>
struct Plus<Raster<U>, Raster<V>>
{

    using A1 = Raster<U>;
    using A2 = Raster<V>;
    using value_type = typename Plus<U, V>::result_type;
    using result_type = Raster<value_type>;
    // using Category = typename Plus<U, V>::Category;
    using Arity = typename Plus<U, V>::Arity;

    result_type operator()(
        Raster<U> const& argument1,
        Raster<V> const& argument2) const
    {
        MaskedArray<U, 2> const& raster1(argument1.value);
        MaskedArray<V, 2> const& raster2(argument2.value);

        assert(raster1.shape()[0] == raster2.shape()[0]);
        assert(raster1.shape()[1] == raster2.shape()[1]);

        size_t const nr_rows = raster1.shape()[0];
        size_t const nr_cols = raster1.shape()[1];

        MaskedArray<value_type, 2> result(fern::extents[nr_rows][nr_cols]);

        // TODO

        // We have result and arguments.
        // - Determine block size.
        // - Determine number of threads.
        // - Spawn threads for each block.
        // - Join afterwards and return result.

        // Use algorithm from algorithm library.


        for(size_t r = 0; r < nr_rows; ++r) {
            for(size_t c = 0; c < nr_cols; ++c) {
                if(raster1.mask()[r][c] || raster2.mask()[r][c]) {
                    result.mask()[r][c] = true;
                }
                else {
                    result[r][c] = raster1[r][c] + raster2[r][c];
                }
            }
        }

        // TODO Move the result out.
        return result_type(result);
    }

};


template<
    class U,
    class V>
struct Plus<Operation<Raster<U>>, Raster<V>>
{

    // using A1 = Operation<Raster<U>>;
    using A1 = Raster<U>;
    using A2 = Raster<V>;
    using value_type = typename Plus<U, V>::result_type;
    using result_type = Raster<value_type>;
    using Arity = typename Plus<U, V>::Arity;
    // using Category = typename Plus<U, V>::Category;

    result_type operator()(
        // Operation<Raster<U>> const& argument1,
        Raster<U> const& argument1,
        Raster<V> const& argument2) const
    {
        // std::cout << "yeah" << std::endl;
        // Data data1(evaluate(argument1));
        // Raster<U> const& raster1(boost::get<Raster<U> const&>(data1).value);

        return Plus<Raster<U>, Raster<V>>()(argument1, argument2);


        // assert(raster1.shape()[0] == raster2.shape()[0]);
        // assert(raster1.shape()[1] == raster2.shape()[1]);

        // size_t const nr_rows = raster1.shape()[0];
        // size_t const nr_cols = raster1.shape()[1];

        // MaskedArray<value_type, 2> result(fern::extents[nr_rows][nr_cols]);

        // for(size_t r = 0; r < nr_rows; ++r) {
        //     for(size_t c = 0; c < nr_cols; ++c) {
        //         if(raster1.mask()[r][c] || raster2.mask()[r][c]) {
        //             result.mask()[r][c] = true;
        //         }
        //         else {
        //             result[r][c] = raster1[r][c] + raster2[r][c];
        //         }
        //     }
        // }

        // // TODO Move the result out.
        // return result_type(result);
    }

};

} // namespace expression_tree
} // namespace fern
