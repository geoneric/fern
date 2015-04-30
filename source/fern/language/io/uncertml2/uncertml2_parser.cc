// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/io/uncertml2/uncertml2_parser.h"
#include <cmath>
#include <sstream>
#include "fern/core/exception.h"
#include "fern/language/io/uncertml2/uncertml2-pskel.hxx"


namespace fern {
namespace {

class NormalDistribution_pimpl:
    public uncertml2::NormalDistribution_pskel
{

public:

    void pre()
    {
    }

    void mean(
        std::vector<double> const& values)
    {
        _means = values;
    }

    void variance(
        std::vector<double> const& values)
    {
        _variances = values;
    }

    std::shared_ptr<fern::NormalDistribution<double> >
        post_NormalDistribution()
    {
        assert(_means.size() == 1);
        assert(_variances.size() == 1);
        assert(_variances[0] >= 0.0);
        return std::make_shared<fern::NormalDistribution<double>>(_means[0],
            std::sqrt(_variances[0]));
    }

private:

    std::vector<double> _means;

    std::vector<double> _variances;

};


class ContinuousValuesType_pimpl:
    public uncertml2::ContinuousValuesType_pskel
{

public:

    void pre()
    {
    }

    void item(
        double value)
    {
        _values.emplace_back(value);
    }

    std::vector<double> const& post_ContinuousValuesType()
    {
        return _values;
    }

private:

    std::vector<double> _values;

};


class PositiveRealValuesType_pimpl:
    public uncertml2::PositiveRealValuesType_pskel
{

public:

    void pre()
    {
        std::cout << "PositiveRealValuesType_pimpl::pre" << std::endl;
    }

    // void item(
    //     double const& value)
    // {
    //     std::cout << "PositiveRealValuesType_pimpl::item1" << " " << value << std::endl;
    //     _values.emplace_back(value);
    // }

    void item(
        double value)
    {
        assert(value > 0.0);
        _values.emplace_back(value);
    }

    std::vector<double> const& post_PositiveRealValuesType()
    {
        return _values;
    }

private:

    std::vector<double> _values;

};


// class PositiveRealNumber_pimpl:
//     public uncertml2::positiveRealNumber_pskel
// {
// 
// public:
// 
//     void pre()
//     {
//         std::cout << "PositiveRealNumber_pimpl::pre" << std::endl;
//     }
// 
//     double post_double()
//     {
//         // TODO Update schema to allow parsing.
//         std::cout << "post_double" << std::endl;
//         // return post_positiveRealNumber();
//         // return uncertml2::positiveRealNumber_pskel::post_positiveRealNumber();
//         return 5.0;
//     }
// 
// private:
// 
// };

} // Anonymous namespace

//! Parse the Xml in \a stream and return an uncertainty instance.
/*!
  \param     stream Stream with Xml to parse.
  \return    Root of Uncertainty specialization.
  \exception std::exception In case of a System category error.
  \exception xml_schema::parsing In case of Xml category error.
*/
std::shared_ptr<Uncertainty> UncertML2Parser::parse(
    std::istream& stream) const
{
    xml_schema::double_pimpl double_p;
    xml_schema::uri_pimpl uri_p;
    xml_schema::string_pimpl string_p;

    ContinuousValuesType_pimpl continuous_values_p;
    continuous_values_p.parsers(double_p, uri_p, string_p);

    // PositiveRealNumber_pimpl positive_real_number_p;

    PositiveRealValuesType_pimpl positive_real_values_p;
    // positive_real_values_p.parsers(positive_real_number_p, uri_p, string_p);
    positive_real_values_p.parsers(double_p, uri_p, string_p);

    NormalDistribution_pimpl normal_distribution_p;
    normal_distribution_p.parsers(continuous_values_p, positive_real_values_p);

    xml_schema::document doc_p(normal_distribution_p,
        "http://www.uncertml.org/2.0", "NormalDistribution", true);

    normal_distribution_p.pre();
    doc_p.parse(stream);
    return normal_distribution_p.post_NormalDistribution();
}


/*!
  \overload
  \param     xml String with Xml to parse.
*/
std::shared_ptr<Uncertainty> UncertML2Parser::parse(
    std::string const& xml) const
{
    // Copy string contents in a string stream and work with that.
    std::stringstream stream;
    stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    stream << xml; // << std::endl;

    std::shared_ptr<Uncertainty> vertex;

    try {
        vertex = parse(stream);
    }
    catch(xml_schema::parsing const& exception) {
        assert(!exception.diagnostics().empty());
        std::cout
            << exception.diagnostics()[0].line()
            << " "
            << exception.diagnostics()[0].column()
            << " "
            << exception.diagnostics()[0].message()
            << std::endl;
        BOOST_THROW_EXCEPTION(fern::detail::ParseError()
            << detail::ExceptionSourceName("<string>")
            << detail::ExceptionLineNr(exception.diagnostics()[0].line())
            << detail::ExceptionColNr(exception.diagnostics()[0].column())
            << detail::ExceptionMessage(exception.diagnostics()[0].message())
        );
    }

    return vertex;
}

} // namespace fern
