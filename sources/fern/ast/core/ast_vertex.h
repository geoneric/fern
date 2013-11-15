#pragma once
#include <memory>
#include <vector>
#include <loki/Visitor.h>
#include "fern/core/string.h"


namespace fern {

class ExpressionVertex;
class StatementVertex;

typedef std::vector<std::shared_ptr<ExpressionVertex>> ExpressionVertices;
typedef std::vector<std::shared_ptr<StatementVertex>> StatementVertices;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AstVertex:
    public Loki::BaseVisitable<>
{

private:

    typedef std::vector<AstVertex*> AstVertices;

public:

    LOKI_DEFINE_VISITABLE()

    typedef AstVertices::size_type size_type;

    virtual        ~AstVertex          ()=default;

                   AstVertex           (AstVertex&&)=delete;

    AstVertex&     operator=           (AstVertex&&)=delete;

                   AstVertex           (AstVertex const&)=delete;

    AstVertex&     operator=           (AstVertex const&)=delete;

    void           set_position        (int line_nr,
                                        int col_id);

    int            line                () const;

    int            col                 () const;

    bool           has_successor       () const;

    AstVertices const& successors      () const;

    AstVertex const* successor         () const;

    AstVertex*     successor           ();

    AstVertex const* successor         (size_type index) const;

    AstVertex*     successor           (size_type index);

    void           add_successor       (AstVertex* successor);

    void           set_successor       (AstVertex* successor);

protected:

                   AstVertex           ();

                   AstVertex           (int line_nr,
                                        int col_id);

private:

    int            _line;

    int            _col;

    //! The next vertex/vertices to process.
    AstVertices    _successors;

};

} // namespace fern
