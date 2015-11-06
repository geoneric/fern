// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern operation_xml parser
#include <boost/test/unit_test.hpp>
#include "fern/core/exception.h"
#include "fern/language/io/uncertml2/uncertml2_parser.h"
#include "fern/language/uncertainty/normal_distribution.h"


BOOST_AUTO_TEST_CASE(parse)
{
    fern::UncertML2Parser parser;
    std::string xml;
    std::shared_ptr<fern::Uncertainty> uncertainty;

    // Example from the uncertml reference.
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
        std::shared_ptr<fern::NormalDistribution<double>>
            normal_distribution(
                std::dynamic_pointer_cast<fern::NormalDistribution<double>>(
                uncertainty));
        BOOST_REQUIRE(normal_distribution);
        BOOST_CHECK_CLOSE(normal_distribution->mean(), 3.14, 0.001);
        BOOST_CHECK_CLOSE(normal_distribution->standard_deviation(),
            std::sqrt(3.14), 0.001);
    }

    // Empty file.
    {
        xml = "";
        BOOST_CHECK_THROW(parser.parse(xml), fern::detail::ParseError);
    }

    // Parse error.
    {
        xml = "abc";
        BOOST_CHECK_THROW(parser.parse(xml), fern::detail::ParseError);
    }
}
