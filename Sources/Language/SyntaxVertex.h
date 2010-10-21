#ifndef INCLUDED_RANALLY_SYNTAXVERTEX
#define INCLUDED_RANALLY_SYNTAXVERTEX

#include <loki/Visitor.h>



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SyntaxVertex: public Loki::BaseVisitable<>
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
