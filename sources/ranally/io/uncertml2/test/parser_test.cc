#define BOOST_TEST_MODULE ranally operation_xml
#include <boost/test/unit_test.hpp>
#include "ranally/io/uncertml2/uncertml2_parser.h"
#include "ranally/uncertainty/normal_distribution.h"


BOOST_AUTO_TEST_SUITE(io_uncertml2_parser)

BOOST_AUTO_TEST_CASE(parse)
{
    ranally::UncertML2Parser parser;
    ranally::String xml;
    std::shared_ptr<ranally::Uncertainty> uncertainty;

    {
        xml =
            "<?xml version=\"1.0\"?>\n"
            "<un:NormalDistribution xmlns:un=\"http://www.uncertml.org/2.0\">\n"
              "<un:mean>3.14</un:mean>\n"
              "<un:variance>3.14</un:variance>\n"
            "</un:NormalDistribution>\n"
            ;
        uncertainty = parser.parse(xml);
        BOOST_REQUIRE(uncertainty);
        std::shared_ptr<ranally::NormalDistribution<double>>
            normal_distribution(
                std::dynamic_pointer_cast<ranally::NormalDistribution<double>>(
                uncertainty));
        BOOST_REQUIRE(normal_distribution);
        BOOST_CHECK_CLOSE(normal_distribution->mean(), 3.14, 0.001);
        BOOST_CHECK_CLOSE(normal_distribution->standard_deviation(),
            std::sqrt(3.14), 0.001);
    }
}

BOOST_AUTO_TEST_SUITE_END()
