#ifndef INCLUDED_RANALLY_SYNTAXVERTEX
#define INCLUDED_RANALLY_SYNTAXVERTEX

#include <vector>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <loki/Visitor.h>
#include <unicode/unistr.h>



namespace ranally {

class ExpressionVertex;
class StatementVertex;

typedef std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
  ExpressionVertices;
typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
  StatementVertices;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SyntaxVertex: private boost::noncopyable,
                    public Loki::BaseVisitable<UnicodeString>
{

  friend class SyntaxVertexTest;

private:

  int              _line;

  int              _col;

protected:

                   SyntaxVertex        ();

                   SyntaxVertex        (int lineNr,
                                        int colId);

public:

  LOKI_DEFINE_VISITABLE()

  virtual          ~SyntaxVertex       ();

  void             setPosition         (int lineNr,
                                        int colId);

  int              line                () const;

  int              col                 () const;

};

} // namespace ranally

#endif
