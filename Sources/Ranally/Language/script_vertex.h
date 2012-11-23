#pragma once
#include "Ranally/Language/statement_vertex.h"


namespace ranally {

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

                   ~ScriptVertex       ()=default;

                   ScriptVertex        (ScriptVertex&&)=delete;

    ScriptVertex&  operator=           (ScriptVertex&&)=delete;

                   ScriptVertex        (ScriptVertex const&)=delete;

    ScriptVertex&  operator=           (ScriptVertex const&)=delete;

    String const&  sourceName          () const;

    StatementVertices const& statements  () const;

    StatementVertices& statements        ();

private:

    String         _sourceName;

    StatementVertices _statements;

};

typedef std::shared_ptr<ScriptVertex> ScriptVertexPtr;

// bool               operator==          (ScriptVertex const& lhs,
//                                         ScriptVertex const& rhs);
// 
// bool               operator!=          (ScriptVertex const& lhs,
//                                         ScriptVertex const& rhs);

} // namespace ranally
