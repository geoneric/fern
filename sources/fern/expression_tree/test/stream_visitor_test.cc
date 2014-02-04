#define BOOST_TEST_MODULE fern expression_tree stream_visitor
#include <sstream>
#include <boost/test/unit_test.hpp>
#include "fern/expression_tree/plus.h"
#include "fern/expression_tree/slope.h"
#include "fern/expression_tree/times.h"
#include "fern/expression_tree/stream_visitor.h"


BOOST_AUTO_TEST_SUITE(stream_visitor)

BOOST_AUTO_TEST_CASE(stream_visitor)
{
    // Never visit an empty tree! Because of the never empty guarantee of
    // boost::variant, the root expression instance always has a value (the
    // first value in the variant).
    // {
    //     fern::Expression expression;
    //     std::stringstream stream;
    //     boost::apply_visitor(fern::StreamVisitor(stream), expression);
    //     BOOST_CHECK_EQUAL(stream.str(), "");
    // }

    // 4
    {
        fern::Constant<int32_t> constant(4);
        auto expression(constant);

        std::stringstream string_stream;
        fern::stream(expression, string_stream);
        BOOST_CHECK_EQUAL(string_stream.str(),
            "(int32_t(4))");
    }

    // 4 + 5
    {
        fern::Constant<int32_t> expression1(4);
        typedef decltype(expression1)::result_type Result1;

        fern::Constant<int64_t> expression2(5);
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

        std::stringstream string_stream;
        fern::stream(expression3, string_stream);
        BOOST_CHECK_EQUAL(string_stream.str(),
            "(plus(int32_t(4), int64_t(5)))");
    }

    // slope((4 + 5) * 3.4)
    {
        // 4
        fern::Constant<int32_t> expression1(4);
        typedef decltype(expression1)::result_type Result1;

        // 5
        fern::Constant<int64_t> expression2(5);
        typedef decltype(expression2)::result_type Result2;

        // 4 + 5
        // TODO To allow for the compiler to perform compile time checks,
        //      type information must be available at compile time. This
        //      means, that all expression types must be able to present their
        //      result type. Operations must be able to tell what they will
        //      output.
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

        // 3.4
        fern::Constant<double> expression4(3.4);
        typedef decltype(expression4)::result_type Result4;
        typedef typename fern::Times<Result3, Result4>::result_type Result5;

        // (4 + 5) * 3.4
        fern::Operation<Result5> expression5(
            "times",
            fern::Implementation(
                fern::Times<Result3, Result4>()),
            {
                expression3,
                expression4
            }
        );

        // slope((4 + 5) * 3.4)
        typedef typename fern::Slope<Result5>::result_type Result6;
        fern::Operation<Result6> expression6(
            "slope",
            fern::Implementation(
                fern::Slope<Result5>()),
            {
                expression5
            }
        );

        std::stringstream string_stream;
        fern::stream(expression6, string_stream);
        BOOST_CHECK_EQUAL(string_stream.str(),
            "(slope(times(plus(int32_t(4), int64_t(5)), double(3.4))))");
    }
}

BOOST_AUTO_TEST_SUITE_END()
