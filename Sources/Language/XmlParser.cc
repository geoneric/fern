#include "XmlParser.h"

#include <sstream>

#include "dev_UnicodeUtils.h"

#include "Ranally-pskel.hxx"



namespace {

class Ranally_pimpl: public ranally::Ranally_pskel
{
public:
  void Expression()
  {
  }
};



class Expression_pimpl: public ranally::Expression_pskel
{
private:
  unsigned long long _line;
  unsigned long long _col;
  std::string      _name;

public:
  void line(
    unsigned long long line)
  {
    _line = line;
  }

  void col(
    unsigned long long col)
  {
    _col = col;
  }

  void Name(
    std::string const& name)
  {
    _name = name;
  }
};

} // Anonymous namespace



namespace ranally {

XmlParser::XmlParser()

  : dev::XercesClient()

{
  assert(dev::XercesClient::isInitialized());
}



XmlParser::~XmlParser()
{
}



SyntaxTree XmlParser::parse(
         std::istream& stream) const
{
  xml_schema::string_pimpl string_p;
  xml_schema::non_negative_integer_pimpl non_negative_integer_p;

  Expression_pimpl expression_p;
  expression_p.parsers(string_p, non_negative_integer_p,
    non_negative_integer_p);

  Ranally_pimpl ranally_p;
  ranally_p.parsers(expression_p);

  xml_schema::document doc_p(ranally_p, "Ranally");

  ranally_p.pre();
  doc_p.parse(stream);
  ranally_p.post_Ranally();

  return 5;
}



//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception std::exception In case of a System category error.
  \exception xml_schema::parsing In case of Xml category error.
  \warning   .
  \sa        .
*/
SyntaxTree XmlParser::parse(
         UnicodeString const& xml) const
{
  // Copy string contents in a string stream and work with that.
  std::stringstream stream;
  stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
  stream << dev::encodeInUTF8(xml); // << std::endl;

  return parse(stream);
}

} // namespace ranally

