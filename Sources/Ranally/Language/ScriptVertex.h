#pragma once
#include "Ranally/Language/StatementVertex.h"


namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ScriptVertex:
    public SyntaxVertex
{

    friend class ScriptVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   ScriptVertex        (String const& sourceName,
                                        StatementVertices const& statements);

  //                  ScriptVertex        (ScriptVertex const& other);

                   ~ScriptVertex       ();

    String const&  sourceName          () const;

    StatementVertices const& statements  () const;

    StatementVertices& statements        ();

private:

    String         _sourceName;

    StatementVertices _statements;

};

typedef boost::shared_ptr<ScriptVertex> ScriptVertexPtr;

// bool               operator==          (ScriptVertex const& lhs,
//                                         ScriptVertex const& rhs);
// 
// bool               operator!=          (ScriptVertex const& lhs,
//                                         ScriptVertex const& rhs);

} // namespace language
} // namespace ranally
