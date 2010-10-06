#include "XmlParser.h"

#include <sstream>
#include <boost/make_shared.hpp>

#include "dev_UnicodeUtils.h"

#include "Ranally-pskel.hxx"

#include "NameVertex.h"
#include "StringVertex.h"



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
    boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
  {
    assert(vertex);

    // TODO add vertex to tree;
  }

  void Assignment(
    boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
  {
    assert(vertex);

    // TODO add vertex to tree;
  }

  boost::shared_ptr<ranally::SyntaxTree> post_Ranally()
  {
    return _syntaxTree;
  }
};



class Assignment_pimpl: public ranally::Assignment_pskel
{
private:
  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > _targets;
  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > _expressions;

public:
  void pre()
  {
    _targets.clear();
    _expressions.clear();
  }

  void Targets(
    std::vector<boost::shared_ptr<ranally::ExpressionVertex> > const& vertices)
  {
    assert(!vertices.empty());
    _targets = vertices;
  }

  void Expressions(
    std::vector<boost::shared_ptr<ranally::ExpressionVertex> > const& vertices)
  {
    assert(!vertices.empty());
    _expressions = vertices;
  }

  boost::shared_ptr<ranally::AssignmentVertex> post_Assignment()
  {
    assert(!_targets.empty() && !_expressions.empty());
    return boost::make_shared<ranally::AssignmentVertex>(_targets,
      _expressions);
  }
};



class Targets_pimpl: public ranally::Targets_pskel
{
private:
  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > _vertices;

public:
  void pre()
  {
    _vertices.clear();
  }

  void Expression(
    boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
  {
    assert(vertex);
    _vertices.push_back(vertex);
  }

  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > post_Targets()
  {
    assert(!_vertices.empty());
    return _vertices;
  }
};



class Expressions_pimpl: public ranally::Expressions_pskel
{
private:
  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > _vertices;

public:
  void pre()
  {
    _vertices.clear();
  }

  void Expression(
    boost::shared_ptr<ranally::ExpressionVertex> const& vertex)
  {
    assert(vertex);
    _vertices.push_back(vertex);
  }

  std::vector<boost::shared_ptr<ranally::ExpressionVertex> > post_Expressions()
  {
    assert(!_vertices.empty());
    return _vertices;
  }
};



class Expression_pimpl: public ranally::Expression_pskel
{
private:
  int              _line;
  int              _col;
  UnicodeString    _name;
  UnicodeString    _string;
  boost::shared_ptr<ranally::ExpressionVertex> _vertex;

public:
  void pre()
  {
    _line = -1;
    _col = -1;
    _name = UnicodeString();
    _string = UnicodeString();
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

  void String(
    std::string const& string)
  {
    _string = dev::decodeFromUTF8(string);
    _vertex = boost::make_shared<ranally::StringVertex>(_line, _col, _string);
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
  expression_p.parsers(string_p, string_p, non_negative_integer_p,
    non_negative_integer_p);

  Targets_pimpl targets_p;
  targets_p.parsers(expression_p);

  Expressions_pimpl expressions_p;
  expressions_p.parsers(expression_p);

  Assignment_pimpl assignment_p;
  assignment_p.parsers(targets_p, expressions_p);

  Ranally_pimpl ranally_p;
  ranally_p.parsers(expression_p, assignment_p);

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

