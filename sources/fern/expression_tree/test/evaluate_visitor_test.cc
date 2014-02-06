#define BOOST_TEST_MODULE fern expression_tree evaluate_visitor
#include <boost/multi_array.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/variant/get.hpp>
#include "fern/expression_tree/plus.h"
#include "fern/expression_tree/sqrt.h"
#include "fern/expression_tree/times.h"
#include "fern/expression_tree/evaluate_visitor.h"


BOOST_AUTO_TEST_SUITE(evaluate_visitor)

BOOST_AUTO_TEST_CASE(visit_constants)
{
    // 2
    {
        fern::Constant<int32_t> constant(2);
        auto expression(constant);

        typedef decltype(expression)::result_type Result;

        fern::Data result(fern::evaluate(expression));

        BOOST_REQUIRE_NO_THROW(boost::get<Result>(result));
        BOOST_CHECK_EQUAL(boost::get<Result>(result).value, 2);
    }

    // 2 + 1
    {
        fern::Constant<int32_t> expression1(2);
        typedef decltype(expression1)::result_type Result1;

        fern::Constant<int32_t> expression2(1);
        typedef decltype(expression2)::result_type Result2;

        typedef typename fern::Plus<Result1, Result2>::result_type Result3;

        fern::Operation<Result3> expression3(
            "plus",
            fern::Implementation(
                fern::Plus<Result1, Result2>()),
            {
                expression1,
                expression2
            }
        );

        fern::Data result(fern::evaluate(expression3));

        BOOST_REQUIRE_NO_THROW(boost::get<Result3>(result));
        BOOST_CHECK_EQUAL(boost::get<Result3>(result).value, 3);
    }

    // sqrt((2 + 1) * 3.0)
    {
        // 1: 2
        fern::Constant<int32_t> expression1(2);
        typedef decltype(expression1)::result_type Result1;

        // 2: 1
        fern::Constant<int32_t> expression2(1);
        typedef decltype(expression2)::result_type Result2;

        // 3: 2 + 1
        typedef typename fern::Plus<Result1, Result2>::result_type Result3;

        fern::Operation<Result3> expression3(
            "plus",
            fern::Implementation(
                fern::Plus<Result1, Result2>()),
            {
                expression1,
                expression2
            }
        );

        // 4: 3.0
        fern::Constant<double> expression4(3.0);
        typedef decltype(expression4)::result_type Result4;

        // 5: (2 + 1) * 3.0
        typedef typename fern::Times<Result3, Result4>::result_type Result5;

        fern::Operation<Result5> expression5(
            "times",
            fern::Implementation(
                fern::Times<Result3, Result4>()),
            {
                expression3,
                expression4
            }
        );

        // 6: sqrt((2 + 1) * 3.0)
        typedef typename fern::Sqrt<Result5>::result_type Result6;

        fern::Operation<Result6> expression6(
            "sqrt",
            fern::Implementation(
                fern::Sqrt<Result5>()),
            {
                expression5
            }
        );

        fern::Data result(fern::evaluate(expression6));

        BOOST_REQUIRE_NO_THROW(boost::get<Result6>(result));
        BOOST_CHECK_CLOSE(boost::get<Result6>(result).value, 3.0, 1e-6);
    }
}


BOOST_AUTO_TEST_CASE(visit_raster)
{
    size_t const nr_rows = 3;
    size_t const nr_cols = 4;

    // raster + raster
    {
        typedef boost::multi_array<int32_t, 2> MultiArray;
        auto extents(boost::extents[nr_rows][nr_cols]);

        MultiArray array1(extents);
        array1[0][0] = -2;
        array1[0][1] = -1;
        array1[1][0] = 0;
        array1[1][1] = 9;
        array1[2][0] = 1;
        array1[2][1] = 2;

        fern::Array<int32_t> expression1(array1);
        typedef decltype(expression1)::result_type Result1;

        MultiArray array2(extents);
        array2[0][0] = -2;
        array2[0][1] = -1;
        array2[1][0] = 0;
        array2[1][1] = 9;
        array2[2][0] = 1;
        array2[2][1] = 2;

        fern::Array<int32_t> expression2(array2);
        typedef decltype(expression2)::result_type Result2;

        typedef typename fern::Plus<Result1, Result2>::result_type Result3;

        fern::Operation<Result3> expression3(
            "plus",
            fern::Implementation(
                fern::Plus<Result1, Result2>()),
            {
                expression1,
                expression2
            }
        );

        fern::Data result(fern::evaluate(expression3));

        BOOST_REQUIRE_NO_THROW(boost::get<Result3 const&>(result));
        // Result3 const& result_array(boost::get<Result3 const&>(result));

        // // Result3 is an Array<int32_t>. We need to get the multi_array
        // // from its guts.
        // typedef boost::multi_array<Result3::value_type, 2> MultiArrayResult;

        // // TODO Visit the tree and obtain the layered data.
        // MultiArrayResult array3(fern::result<MultiArrayResult>(expression3));

        // fern::assign(expression3, array3);

        // BOOST_REQUIRE_EQUAL(result_array.num_dimensions(), 2);
        // BOOST_REQUIRE_EQUAL(result_array.shape()[0], 2);
        // BOOST_REQUIRE_EQUAL(result_array.shape()[1], 3);

        // BOOST_CHECK_EQUAL(result_array[0][0], -4);
        // BOOST_CHECK_EQUAL(result_array[0][1], -2);
        // BOOST_CHECK_EQUAL(result_array[1][0],  0);
        // BOOST_CHECK_EQUAL(result_array[1][1],  18);
        // BOOST_CHECK_EQUAL(result_array[2][0],  2);
        // BOOST_CHECK_EQUAL(result_array[2][1],  4);
    }
}


BOOST_AUTO_TEST_CASE(visit_vector)
{
    {
        /// std::vector<int32_t> vector1({1, 2, 3, 4, 5});
        /// fern::Array<int32_t> expression1(vector1);
        /// typedef decltype(expression1)::result_type Result1;

        /// std::vector<int32_t> vector2({10, 11, 12, 13, 14});
        /// fern::Array<int32_t> expression2(vector2);
        /// typedef decltype(expression2)::result_type Result2;

        /// typedef typename fern::Plus<Result1, Result2>::result_type Result3;

        /// fern::Operation<Result3> expression3(
        ///     "plus",
        ///     fern::Implementation(
        ///         fern::Plus<Result1, Result2>()),
        ///     {
        ///         expression1,
        ///         expression2
        ///     }
        /// );

        // TODO How to create the resulting collection. Do it in the evaluate?
        //      We need the type of the result and the extent(s) of the
        //      input(s).

        // fern::Data result(fern::evaluate(expression3));

        // BOOST_REQUIRE_NO_THROW(boost::get<Result3>(result));
        // BOOST_CHECK_EQUAL(boost::get<Result3>(result).collection,
        //     std::vector<int32_t>({1, 2, 3, 4, 6}));
    }
}


// BOOST_AUTO_TEST_CASE(visit_boost_multi_array)
// {
//     // 2 + 1
//     {
//         typedef boost::multi_array<int32_t, 2> Array;
//         // typedef Array::index Index;
//         auto extents(boost::extents[30000][40000]);
// 
//         Array array1(extents);
//         fern::Array<int32_t> expression1(array1);
//         typedef decltype(expression1)::result_type Result1;
// 
//         Array array2(extents);
//         fern::Array<int32_t> expression2(array2);
//         typedef decltype(expression2)::result_type Result2;
// 
//         typedef typename fern::Plus<Result1, Result2>::result_type Result3;
// 
//         fern::Operation<Result3> expression3(
//             "plus",
//             fern::Implementation(
//                 fern::Plus<Result1, Result2>()),
//             {
//                 expression1,
//                 expression2
//             }
//         );
// 
//         // fern::Data result(fern::evaluate(expression3));
// 
//         // BOOST_REQUIRE_NO_THROW(boost::get<Result3>(result));
//         // BOOST_CHECK_EQUAL(boost::get<Result3>(result).value, 3);
//     }
// }

BOOST_AUTO_TEST_SUITE_END()
