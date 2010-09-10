#include "XmlParser.h"

#include <sstream>
#include <boost/make_shared.hpp>

#include "dev_UnicodeUtils.h"

#include "Ranally-pskel.hxx"

#include "NameVertex.h"



namespace {

class Ranally_pimpl: public ranally::Ranally_pskel
{
private:
  boost::shared_ptr<ranally::SyntaxTree> _syntaxTree;

public:
  void pre()
  {
    _syntaxTree = boost::make_shared<ranally::SyntaxTree>();
  }

  void Expression(
    boost::shared_ptr<ranally::ExpressionVertex> vertex)
  {
    assert(_syntaxTree);
    assert(vertex);

    // TODO add vertex to tree;
  }

  boost::shared_ptr<ranally::SyntaxTree> post_Ranally()
  {
    return _syntaxTree;
  }
};



class Expression_pimpl: public ranally::Expression_pskel
{
private:
  int              _line;
  int              _col;
  UnicodeString    _name;
  boost::shared_ptr<ranally::ExpressionVertex> _vertex;

public:
  void pre()
  {
    _line = -1;
    _col = -1;
    _name = UnicodeString();
  }

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
    _name = dev::decodeFromUTF8(name);
    _vertex = boost::make_shared<ranally::NameVertex>(_line, _col, _name);
  }

  boost::shared_ptr<ranally::ExpressionVertex> post_Expression()
  {
    return _vertex;
  }
};

} // Anonymous namespace



namespace ranally {

XmlParser::XmlParser()
{
}



XmlParser::~XmlParser()
{
}



boost::shared_ptr<SyntaxTree> XmlParser::parse(
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
  return ranally_p.post_Ranally();
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
boost::shared_ptr<SyntaxTree> XmlParser::parse(
         UnicodeString const& xml) const
{
  // Copy string contents in a string stream and work with that.
  std::stringstream stream;
  stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
  stream << dev::encodeInUTF8(xml); // << std::endl;

  return parse(stream);
}

} // namespace ranally

