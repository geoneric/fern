#ifndef INCLUDED_RANALLY_SCRIPTVERTEX
#define INCLUDED_RANALLY_SCRIPTVERTEX

#include "StatementVertex.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ScriptVertex: public SyntaxVertex
{

  friend class ScriptVertexTest;

public:

  LOKI_DEFINE_VISITABLE()

private:

  StatementVertices _statements;

protected:

public:

                   ScriptVertex        (StatementVertices const& statements);

  /* virtual */    ~ScriptVertex       ();

  StatementVertices const& statements  () const;

};

} // namespace language
} // namespace ranally

#endif
