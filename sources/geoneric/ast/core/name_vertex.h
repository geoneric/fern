#pragma once
#include "geoneric/ast/core/expression_vertex.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class NameVertex:
    public ExpressionVertex
{

    friend class NameVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   NameVertex          (String const& name);

                   NameVertex          (int lineNr,
                                        int colId,
                                        String const& name);

                   ~NameVertex         ()=default;

                   NameVertex          (NameVertex&&)=delete;

    NameVertex&    operator=           (NameVertex&&)=delete;

                   NameVertex          (NameVertex const&)=delete;

    NameVertex&    operator=           (NameVertex const&)=delete;

    void           add_definition      (NameVertex* vertex);

    std::vector<NameVertex*> const& definitions() const;

    void           add_use             (NameVertex* vertex);

    std::vector<NameVertex*> const& uses () const;

private:

    // //! Definition of the name (left side of an assignment).
    // NameVertex*    _definition;

    //! Definitions of the name. Only relevant for use vertices.
    std::vector<NameVertex*> _definitions;

    //! Uses of the name in expressions. Only relevant for definition vertices.
    std::vector<NameVertex*> _uses;

};

typedef std::shared_ptr<NameVertex> NameVertexPtr;

inline std::ostream& operator<<(
    std::ostream& stream,
    NameVertex const& vertex)
{
    stream << "name: " << vertex.name().encode_in_utf8() << "\n";

    if(!vertex.definitions().empty()) {
        stream << "definitions:\n";

        for(auto definition: vertex.definitions()) {
            stream << *definition;
        }
    }

    if(!vertex.uses().empty()) {
        stream << "uses:\n";

        for(auto use: vertex.uses()) {
            stream << *use;
        }
    }

    return stream;
}

} // namespace geoneric
