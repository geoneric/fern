#ifndef INCLUDED_RANALLY_SCRIPTVERTEX
#define INCLUDED_RANALLY_SCRIPTVERTEX

#include <vector>
#include <boost/shared_ptr.hpp>

#include "StatementVertex.h"



namespace ranally {

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

  typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
    StatementVertices;

private:

  StatementVertices _statements;

protected:

public:

                   ScriptVertex        (StatementVertices const& statements);

  /* virtual */    ~ScriptVertex       ();

};

} // namespace ranally

#endif
