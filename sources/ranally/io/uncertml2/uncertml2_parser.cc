#include "ranally/io/uncertml2/uncertml2_parser.h"
#include <cmath>
#include <sstream>
// #include <stack>
#include "ranally/core/exception.h"
#include "ranally/io/uncertml2/uncertml2-pskel.hxx"


namespace ranally {
namespace {

class NormalDistribution_pimpl:
    public uncertml2::NormalDistribution_pskel
{

public:

    void pre()
    {
        std::cout << "NormalDistribution_pimpl::pre" << std::endl;
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

    std::shared_ptr<ranally::NormalDistribution<double> >
        post_NormalDistribution()
    {
        std::cout << "post_NormalDistribution" << std::endl;
        assert(_means.size() == 1);
        assert(_variances.size() == 1);
        assert(_variances[0] >= 0.0);
        return std::make_shared<ranally::NormalDistribution<double>>(_means[0],
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
        std::cout << "ContinuousValuesType_pimpl::pre" << std::endl;
    }

    void item(
        double value)
    {
        _values.push_back(value);
    }

    std::vector<double> const& post_ContinuousValuesType()
    {
        std::cout << "post_ContinuousValuesType" << std::endl;
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

    void item(
        double value)
    {
        _values.push_back(value);
    }

    std::vector<double> const& post_PositiveRealValuesType()
    {
        std::cout << "post_PositiveRealValuesType" << std::endl;
        return _values;
    }

private:

    std::vector<double> _values;

};


class PositiveRealNumber_pimpl:
    public uncertml2::positiveRealNumber_pskel
{

public:

    void pre()
    {
        std::cout << "PositiveRealNumber_pimpl::pre" << std::endl;
    }

    double post_double()
    {
        std::cout << "post_double" << std::endl;
        return post_positiveRealNumber();
    }

private:

};

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
    // // Python stores regular integers in a C long. Xsd doesn't have a long
    // // parser, but does have an int parser. Let's make sure a long is of the
    // // same size as an int. Xsd's long parser uses long long, which is good
    // // for Pythons long integer type.
    // // TODO Use long_p for integer parsing? int is too small I would think.
    // // assert(sizeof(int) == sizeof(long));

    // xml_schema::positive_integer_pimpl positive_integer_p;
    // xml_schema::int_pimpl int_p;
    // xml_schema::non_negative_integer_pimpl non_negative_integer_p;
    // xml_schema::long_pimpl long_p;
    // xml_schema::double_pimpl double_p;
    // xml_schema::string_pimpl string_p;

    // Integer_pimpl integer_p;
    // integer_p.parsers(positive_integer_p, int_p);

    // UnsignedInteger_pimpl unsigned_integer_p;
    // unsigned_integer_p.parsers(positive_integer_p, non_negative_integer_p);

    // Float_pimpl float_p;
    // float_p.parsers(positive_integer_p, double_p);

    // Number_pimpl number_p;
    // number_p.parsers(integer_p, unsigned_integer_p, float_p);

    // Expression_pimpl expression_p;
    // // TODO Set parsers?

    // Expressions_pimpl expressions_p;
    // expressions_p.parsers(expression_p);

    // Function_pimpl function_p;
    // function_p.parsers(string_p, expressions_p);

    // Operator_pimpl operator_p;
    // operator_p.parsers(string_p, expressions_p);

    // Subscript_pimpl subscript_p;
    // subscript_p.parsers(expression_p);

    // expression_p.parsers(string_p, subscript_p, string_p, number_p,
    //     function_p, operator_p, non_negative_integer_p,
    //     non_negative_integer_p);

    // // Targets_pimpl targets_p;
    // // targets_p.parsers(expression_p);

    // Assignment_pimpl assignment_p;
    // assignment_p.parsers(expression_p);

    // Statements_pimpl statements_p;

    // If_pimpl if_p;
    // if_p.parsers(expression_p, statements_p);

    // While_pimpl while_p;
    // while_p.parsers(expression_p, statements_p);

    // Statement_pimpl statement_p;
    // statement_p.parsers(expression_p, assignment_p, if_p, while_p,
    //     non_negative_integer_p, non_negative_integer_p);

    // statements_p.parsers(statement_p);

    std::cout << "parse!" << std::endl;

    xml_schema::double_pimpl double_p;
    xml_schema::uri_pimpl uri_p;
    xml_schema::string_pimpl string_p;

    ContinuousValuesType_pimpl continuous_values_p;
    // continuous_values_p.item_parser(double_p);
    // TODO Why doesn't this work???
    // continuous_values_p.parsers(double_p);
    continuous_values_p.parsers(double_p, uri_p, string_p);

    PositiveRealNumber_pimpl positive_real_number_p;

    PositiveRealValuesType_pimpl positive_real_values_p;
    // TODO Why doesn't this work???
    // positive_real_values_p.parsers(positive_real_number_p);
    // positive_real_values_p.item_parser(positive_real_number_p);
    positive_real_values_p.parsers(positive_real_number_p, uri_p, string_p);

    NormalDistribution_pimpl normal_distribution_p;
    normal_distribution_p.parsers(continuous_values_p, positive_real_values_p);
    // normal_distribution_p.mean_parser(continuous_values_p);

    xml_schema::document doc_p(normal_distribution_p, "Blah");

    normal_distribution_p.pre();
    doc_p.parse(stream);
    std::cout << "parse!" << std::endl;
    return normal_distribution_p.post_NormalDistribution();
}


/*!
  \overload
  \param     xml String with Xml to parse.
*/
std::shared_ptr<Uncertainty> UncertML2Parser::parse(
    String const& xml) const
{
    // Copy string contents in a string stream and work with that.
    std::stringstream stream;
    stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    stream << xml.encode_in_utf8(); // << std::endl;

    std::shared_ptr<Uncertainty> vertex;

    try {
        vertex = parse(stream);
    }
    catch(xml_schema::parsing const& exception) {
        assert(!exception.diagnostics().empty());
        BOOST_THROW_EXCEPTION(ranally::detail::ParseError()
            << detail::ExceptionSourceName("<string>")
            << detail::ExceptionLineNr(exception.diagnostics()[0].line())
            << detail::ExceptionColNr(exception.diagnostics()[0].column())
            << detail::ExceptionMessage(exception.diagnostics()[0].message())
        );
    }

    return vertex;
}

} // namespace ranally
