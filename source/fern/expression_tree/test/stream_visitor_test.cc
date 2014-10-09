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
    namespace fet = fern::expression_tree;

    // Never visit an empty tree! Because of the never empty guarantee of
    // boost::variant, the root expression instance always has a value (the
    // first value in the variant).
    // {
    //     fet::Expression expression;
    //     std::stringstream stream;
    //     boost::apply_visitor(fet::StreamVisitor(stream), expression);
    //     BOOST_CHECK_EQUAL(stream.str(), "");
    // }

    // 4
    {
        fet::Constant<int32_t> constant(4);
        auto expression(constant);

        std::stringstream string_stream;
        fet::stream(expression, string_stream);
        BOOST_CHECK_EQUAL(string_stream.str(),
            "(int32_t(4))");
    }

    // 4 + 5
    {
        fet::Constant<int32_t> expression1(4);
        using Result1 = decltype(expression1)::result_type;

        fet::Constant<int64_t> expression2(5);
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

        std::stringstream string_stream;
        fet::stream(expression3, string_stream);
        BOOST_CHECK_EQUAL(string_stream.str(),
            "(plus(int32_t(4), int64_t(5)))");
    }

    // slope((4 + 5) * 3.4)
    {
        // 4
        fet::Constant<int32_t> expression1(4);
        using Result1 = decltype(expression1)::result_type;

        // 5
        fet::Constant<int64_t> expression2(5);
        using Result2 = decltype(expression2)::result_type;

        // 4 + 5
        // TODO To allow for the compiler to perform compile time checks,
        //      type information must be available at compile time. This
        //      means, that all expression types must be able to present their
        //      result type. Operations must be able to tell what they will
        //      output.
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

        // 3.4
        fet::Constant<double> expression4(3.4);
        using Result4 = decltype(expression4)::result_type;
        using Result5 = typename fet::Times<Result3, Result4>::result_type;

        // (4 + 5) * 3.4
        fet::Operation<Result5> expression5(
            "times",
            fet::Implementation(
                fet::Times<Result3, Result4>()),
            {
                expression3,
                expression4
            }
        );

        // slope((4 + 5) * 3.4)
        using Result6 = typename fet::Slope<Result5>::result_type;
        fet::Operation<Result6> expression6(
            "slope",
            fet::Implementation(
                fet::Slope<Result5>()),
            {
                expression5
            }
        );

        std::stringstream string_stream;
        fet::stream(expression6, string_stream);
        BOOST_CHECK_EQUAL(string_stream.str(),
            "(slope(times(plus(int32_t(4), int64_t(5)), double(3.4))))");
    }
}

BOOST_AUTO_TEST_SUITE_END()
