#ifndef INCLUDED_RANALLY_LANGUAGE_SYNTAXVERTEX
#define INCLUDED_RANALLY_LANGUAGE_SYNTAXVERTEX

#include <vector>
#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>
#include <unicode/unistr.h>
#include <boost/shared_ptr.hpp>



namespace ranally {
namespace language {

class ExpressionVertex;
class StatementVertex;

typedef std::vector<boost::shared_ptr<ExpressionVertex> > ExpressionVertices;
typedef std::vector<boost::shared_ptr<StatementVertex> > StatementVertices;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SyntaxVertex:
  private boost::noncopyable,
  public Loki::BaseVisitable<>
{

  friend class SyntaxVertexTest;

private:

  typedef std::vector<SyntaxVertex*> SyntaxVertices;

public:

  LOKI_DEFINE_VISITABLE()

  typedef SyntaxVertices::size_type size_type;

  virtual          ~SyntaxVertex       ();

  void             setPosition         (int lineNr,
                                        int colId);

  int              line                () const;

  int              col                 () const;

  SyntaxVertices const& successors     () const;

  SyntaxVertex const* successor        () const;

  SyntaxVertex*    successor           ();

  SyntaxVertex const* successor        (size_type index) const;

  SyntaxVertex*    successor           (size_type index);

  void             addSuccessor        (SyntaxVertex* successor);

protected:

                   SyntaxVertex        ();

                   SyntaxVertex        (int lineNr,
                                        int colId);

                   SyntaxVertex        (SyntaxVertex const& other);

private:

  int              _line;

  int              _col;

  //! The next vertex/vertices to process.
  SyntaxVertices   _successors;

};

} // namespace language
} // namespace ranally

#endif
