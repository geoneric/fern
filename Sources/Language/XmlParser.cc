#include "XmlParser.h"

#include <iostream>
#include <sstream>

#include "dev_UnicodeUtils.h"

#include "Ranally-pskel.hxx"



namespace {

class Ranally_pimpl: public ranally::Ranally_pskel
{
public:
  void line(
    long long line)
  {
    std::cout << line << std::endl;
  }

  void col(
    long long col)
  {
    std::cout << col << std::endl;
  }

  void Expression()
  {
    std::cout << "Expression" << std::endl;
  }
};



class Expression_pimpl: public ranally::Expression_pskel
{
public:
  void Name(
    std::string const& name)
  {
    std::cout << name << std::endl;
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
         UnicodeString const& xml)
{
  // Do a validating parse of the xml (only in debug?).

  // Generate a Syntax Tree based on the XML contents.

  // boost::scoped_ptr<xercesc::SAX2XMLReader> parser(
  //        xercesc::XMLReaderFactory::createXMLReader());

  try {
    xml_schema::string_pimpl string_p;
    xml_schema::integer_pimpl integer_p;

    Expression_pimpl expression_p;
    expression_p.parsers(string_p);

    Ranally_pimpl ranally_p;
    ranally_p.parsers(expression_p, integer_p, integer_p);

    xml_schema::document doc_p(ranally_p, "Ranally");

    std::stringstream stream;
    stream.exceptions (std::ifstream::badbit | std::ifstream::failbit);
    stream << dev::encodeInUTF8(xml) << std::endl;

    std::cout << dev::encodeInUTF8(xml) << std::endl;

    ranally_p.pre();
    doc_p.parse(stream);
    ranally_p.post_Ranally();

    // TODO error handling using a handler. The caller must be in control.
  }
  catch(xml_schema::parsing const& exception) {
    std::cout << "parsing" << std::endl;
    std::cerr << exception << std::endl;
  }
  catch(xml_schema::exception const& exception) {
    std::cout << "exception" << std::endl;
    std::cerr << exception << std::endl;
  }
  catch(std::exception const& exception) {
    std::cout << "std::exception" << std::endl;
    std::cerr << exception.what() << std::endl;
  }
  catch(...) {
    std::cout << "errror" << std::endl;
  }

  return 5;
}

} // namespace ranally

