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


BOOST_AUTO_TEST_CASE(visit_boost_multi_array)
{
    {
        std::vector<int32_t> vector1(100);
        fern::Array<int32_t> expression1(vector1);
        // typedef decltype(expression1)::result_type Result1;

        // Array array2(extents);
        // fern::Array<int32_t> expression2(array2);
        // typedef decltype(expression2)::result_type Result2;

        // typedef typename fern::Plus<Result1, Result2>::result_type Result3;

        // fern::Operation<Result3> expression3(
        //     "plus",
        //     fern::Implementation(
        //         fern::Plus<Result1, Result2>()),
        //     {
        //         expression1,
        //         expression2
        //     }
        // );

        // fern::Data result(fern::evaluate(expression3));

        // BOOST_REQUIRE_NO_THROW(boost::get<Result3>(result));
        // BOOST_CHECK_EQUAL(boost::get<Result3>(result).value, 3);
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
