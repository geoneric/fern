#include "ranally/io/uncertml2/uncertml2_parser.h"
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

    std::shared_ptr<ranally::NormalDistribution<double> >
        post_NormalDistribution()
    {
    }

};


class ContinuousValuesType_pimpl:
    public uncertml2::ContinuousValuesType_pskel
{
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

    xml_schema::double_pimpl double_p;

    ContinuousValuesType_pimpl continuous_values_p;
    continuous_values_p.item_parser(double_p);

    // PositiveRealNumber_pimpl positive_real_number_p;

    // PositiveRealValues_pimpl positive_real_values_p;
    // positive_real_values_p.parsers(positive_real_number_p);

    NormalDistribution_pimpl normal_distribution_p;
    // normal_distribution_p.parsers(continuous_values_p /* , positive_real_values_p */);
    normal_distribution_p.mean_parser(continuous_values_p);

    xml_schema::document doc_p(normal_distribution_p, "Blah");

    normal_distribution_p.pre();
    doc_p.parse(stream);
    return normal_distribution_p.post_NormalDistribution();
}


/*!
  \overload
  \param     xml String with Xml to parse.
*/
std::shared_ptr<Uncertainty> UncertML2Parser::parse_string(
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
