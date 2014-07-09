#define BOOST_TEST_MODULE fern expression_tree evaluate_visitor
#include <boost/multi_array.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/variant/get.hpp>
#include "fern/expression_tree/plus.h"
#include "fern/expression_tree/sqrt.h"
#include "fern/expression_tree/times.h"
#include "fern/expression_tree/evaluate_visitor.h"
#include "fern/feature/core/masked_array.h"


namespace fern {
namespace expression_tree {

// template<
//     class U,
//     class V>
// Operation<typename Plus<Raster<U>, Raster<V>>::result_type> operator+(
//     Raster<U> const& lhs,
//     Raster<V> const& rhs)
// {
//     return Operation<typename Plus<Raster<U>, Raster<V>>::result_type>(
//         "plus",
//         Implementation(Plus<Raster<U>, Raster<V>>()),
//         {
//             lhs,
//             rhs
//         }
//     );
// }


// template<
//     class U,
//     class V>
// Operation<typename Plus<Raster<U>, Raster<V>>::result_type> operator/(
//     Operation<Raster<U>> const& lhs,
//     Raster<V> const& rhs)
// {
//     return Operation<typename Plus<Raster<U>, Raster<V>>::result_type>(
//         "plus",
//         Implementation(Plus<Raster<U>, Raster<V>>()),
//         {
//             lhs,
//             rhs
//         }
//     );
// }


template<
    class U,
    class V>
Operation<typename Plus<U, V>::result_type> operator+(
    U const& lhs,
    V const& rhs)
{
    return Operation<typename Plus<U, V>::result_type>(
        "plus",
        Implementation(Plus<U, V>()),
        {
            lhs,
            rhs
        }
    );
}

} // namespace expression_tree
} // namespace fern


BOOST_AUTO_TEST_SUITE(evaluate_visitor)

BOOST_AUTO_TEST_CASE(visit_constants)
{
    namespace fet = fern::expression_tree;

    // 2
    {
        fet::Constant<int32_t> constant(2);
        auto expression(constant);

        using Result = decltype(expression)::result_type;

        fet::Data result(fet::evaluate(expression));

        BOOST_REQUIRE_NO_THROW(boost::get<Result>(result));
        BOOST_CHECK_EQUAL(boost::get<Result>(result).value, 2);
    }

    // 2 + 1
    {
        fet::Constant<int32_t> expression1(2);
        using Result1 = decltype(expression1)::result_type;

        fet::Constant<int32_t> expression2(1);
        using Result2 = decltype(expression2)::result_type;

        using Result3 = typename fet::Plus<Result1, Result2>::result_type;

        fet::Operation<Result3> expression3(
            "plus",
            fet::Implementation(
                fet::Plus<Result1, Result2>()),
            {
                expression1,
                expression2
            }
        );

        fet::Data result(fet::evaluate(expression3));

        BOOST_REQUIRE_NO_THROW(boost::get<Result3>(result));
        BOOST_CHECK_EQUAL(boost::get<Result3>(result).value, 3);
    }

    // sqrt((2 + 1) * 3.0)
    {
        // 1: 2
        fet::Constant<int32_t> expression1(2);
        using Result1 = decltype(expression1)::result_type;

        // 2: 1
        fet::Constant<int32_t> expression2(1);
        using Result2 = decltype(expression2)::result_type;

        // 3: 2 + 1
        using Result3 = typename fet::Plus<Result1, Result2>::result_type;

        fet::Operation<Result3> expression3(
            "plus",
            fet::Implementation(
                fet::Plus<Result1, Result2>()),
            {
                expression1,
                expression2
            }
        );

        // 4: 3.0
        fet::Constant<double> expression4(3.0);
        using Result4 = decltype(expression4)::result_type;

        // 5: (2 + 1) * 3.0
        using Result5 = typename fet::Times<Result3, Result4>::result_type;

        fet::Operation<Result5> expression5(
            "times",
            fet::Implementation(
                fet::Times<Result3, Result4>()),
            {
                expression3,
                expression4
            }
        );

        // 6: sqrt((2 + 1) * 3.0)
        using Result6 = typename fet::Sqrt<Result5>::result_type;

        fet::Operation<Result6> expression6(
            "sqrt",
            fet::Implementation(
                fet::Sqrt<Result5>()),
            {
                expression5
            }
        );

        fet::Data result(fet::evaluate(expression6));

        BOOST_REQUIRE_NO_THROW(boost::get<Result6>(result));
        BOOST_CHECK_CLOSE(boost::get<Result6>(result).value, 3.0, 1e-6);
    }
}


template<
    class T>
using Raster = fern::MaskedArray<T, 2>;


BOOST_AUTO_TEST_CASE(visit_raster)
{
    namespace fet = fern::expression_tree;

    size_t const nr_rows = 3;
    size_t const nr_cols = 4;
    auto extents(fern::extents[nr_rows][nr_cols]);

    Raster<int32_t> raster1(extents);
    raster1[0][0] = -2;
    raster1[0][1] = -1;
    raster1[1][0] = 0;
    raster1.mask()[1][1] = true;
    raster1[2][0] = 1;
    raster1[2][1] = 2;

    auto expression1 = fet::Raster<int32_t>(raster1);
    using Result1 = decltype(expression1)::result_type;
    static_assert(std::is_same<Result1, fet::Raster<int32_t>>::value, "");

    Raster<int32_t> raster2(extents);
    raster2[0][0] = -20;
    raster2[0][1] = -10;
    raster2[1][0] = 0;
    raster2.mask()[1][1] = true;
    raster2[2][0] = 10;
    raster2[2][1] = 20;

    auto expression2 = fet::Raster<int32_t>(raster2);
    using Result2 = decltype(expression2)::result_type;
    static_assert(std::is_same<Result2, fet::Raster<int32_t>>::value, "");

    Raster<double> raster3(extents);
    raster3[0][0] = 2.0;
    raster3[0][1] = 4.0;
    raster3[1][0] = 6.0;
    raster3[1][1] = 8.0;
    raster3[2][0] = 10.0;
    raster3[2][1] = 12.0;

    auto expression3 = fet::Raster<double>(raster3);
    using Result3 = decltype(expression3)::result_type;
    static_assert(std::is_same<Result3, fet::Raster<double>>::value, "");

    // raster + raster
    {
        using namespace fern::expression_tree;

        auto operation = expression1 + expression2;

        fet::evaluate(operation);
        fet::Data result(fet::evaluate(operation));

        BOOST_REQUIRE_NO_THROW(boost::get<fet::Raster<int32_t> const&>(result));
        fet::Raster<int32_t> const& result_raster(
            boost::get<fet::Raster<int32_t> const&>(result));
        Raster<int32_t> const& raster(result_raster.value);

        BOOST_REQUIRE_EQUAL(raster.num_dimensions(), 2);
        BOOST_REQUIRE_EQUAL(raster.shape()[0], nr_rows);
        BOOST_REQUIRE_EQUAL(raster.shape()[1], nr_cols);

        BOOST_CHECK(!raster.mask()[0][0]);
        BOOST_CHECK_EQUAL(raster[0][0], -22);

        BOOST_CHECK(!raster.mask()[0][1]);
        BOOST_CHECK_EQUAL(raster[0][1], -11);

        BOOST_CHECK(!raster.mask()[1][0]);
        BOOST_CHECK_EQUAL(raster[1][0],  0);

        BOOST_CHECK( raster.mask()[1][1]);

        BOOST_CHECK(!raster.mask()[2][0]);
        BOOST_CHECK_EQUAL(raster[2][0],  11);

        BOOST_CHECK(!raster.mask()[2][1]);
        BOOST_CHECK_EQUAL(raster[2][1],  22);
    }

    // raster + raster + raster
    {
        auto operation = expression1 + expression2 + expression3;

        // std::cout << "evaluate..." << std::endl;
        fet::evaluate(operation);
        // std::cout << "/evaluate..." << std::endl;
        // fet::Data result(fet::evaluate(operation));

        // BOOST_REQUIRE_NO_THROW(boost::get<fet::Raster<double> const&>(result));
        // fet::Raster<double> const& result_raster(
        //     boost::get<fet::Raster<double> const&>(result));
        // Raster<double> const& raster(result_raster.value);
        // return;

        // BOOST_REQUIRE_EQUAL(raster.num_dimensions(), 2);
        // BOOST_REQUIRE_EQUAL(raster.shape()[0], nr_rows);
        // BOOST_REQUIRE_EQUAL(raster.shape()[1], nr_cols);

        // BOOST_CHECK(!raster.mask()[0][0]);
        // BOOST_CHECK_EQUAL(raster[0][0], -22);

        // BOOST_CHECK(!raster.mask()[0][1]);
        // BOOST_CHECK_EQUAL(raster[0][1], -11);

        // BOOST_CHECK(!raster.mask()[1][0]);
        // BOOST_CHECK_EQUAL(raster[1][0],  0);

        // BOOST_CHECK( raster.mask()[1][1]);

        // BOOST_CHECK(!raster.mask()[2][0]);
        // BOOST_CHECK_EQUAL(raster[2][0],  11);

        // BOOST_CHECK(!raster.mask()[2][1]);
        // BOOST_CHECK_EQUAL(raster[2][1],  22);
    }
}


BOOST_AUTO_TEST_CASE(visit_vector)
{
    {
        /// std::vector<int32_t> vector1({1, 2, 3, 4, 5});
        /// fern::Array<int32_t> expression1(vector1);
        /// using Result1 = decltype(expression1)::result_type;

        /// std::vector<int32_t> vector2({10, 11, 12, 13, 14});
        /// fern::Array<int32_t> expression2(vector2);
        /// using Result2 = decltype(expression2)::result_type;

        /// using Result3 = typename fern::Plus<Result1, Result2>::result_type;

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
//         using Array = boost::multi_array<int32_t, 2>;
//         // using Index = Array::index;
//         auto extents(boost::extents[30000][40000]);
// 
//         Array array1(extents);
//         fern::Array<int32_t> expression1(array1);
//         using Result1 = decltype(expression1)::result_type;
// 
//         Array array2(extents);
//         fern::Array<int32_t> expression2(array2);
//         using Result2 = decltype(expression2)::result_type;
// 
//         using Result3 = typename fern::Plus<Result1, Result2>::result_type;
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
