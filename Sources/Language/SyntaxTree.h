#ifndef INCLUDED_RANALLY_SYNTAXTREE
#define INCLUDED_RANALLY_SYNTAXTREE

#include <vector>
#include <boost/shared_ptr.hpp>

#include "StatementVertex.h"
#include "SyntaxVertex.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  TODO Rename to ScriptVertex

  \sa        .
*/
class SyntaxTree: public SyntaxVertex
{

  friend class SyntaxTreeTest;

public:

  LOKI_DEFINE_VISITABLE()

  typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
    StatementVertices;

private:

  StatementVertices _statements;

protected:

public:

                   SyntaxTree               (
                                       StatementVertices const& statements);

  /* virtual */    ~SyntaxTree              ();

};

} // namespace ranally

#endif
