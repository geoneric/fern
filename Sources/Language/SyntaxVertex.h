#ifndef INCLUDED_RANALLY_SYNTAXVERTEX
#define INCLUDED_RANALLY_SYNTAXVERTEX



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SyntaxVertex
{

  friend class SyntaxVertexTest;

private:

  int              _line;

  int              _col;

protected:

public:

                   SyntaxVertex        (int lineNr,
                                        int colId);

  virtual          ~SyntaxVertex       ();

  int              line                () const;

  int              col                 () const;

};

} // namespace ranally

#endif
