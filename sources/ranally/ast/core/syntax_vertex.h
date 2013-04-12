#pragma once
#include <memory>
#include <vector>
#include <loki/Visitor.h>
#include "ranally/core/string.h"


namespace ranally {

class ExpressionVertex;
class StatementVertex;

typedef std::vector<std::shared_ptr<ExpressionVertex>> ExpressionVertices;
typedef std::vector<std::shared_ptr<StatementVertex>> StatementVertices;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SyntaxVertex:
    public Loki::BaseVisitable<>
{

    friend class SyntaxVertexTest;

private:

    typedef std::vector<SyntaxVertex*> SyntaxVertices;

public:

    LOKI_DEFINE_VISITABLE()

    typedef SyntaxVertices::size_type size_type;

    virtual        ~SyntaxVertex       ()=default;

                   SyntaxVertex        (SyntaxVertex&&)=delete;

    SyntaxVertex&  operator=           (SyntaxVertex&&)=delete;

                   SyntaxVertex        (SyntaxVertex const&)=delete;

    SyntaxVertex&  operator=           (SyntaxVertex const&)=delete;

    void           set_position        (int line_nr,
                                        int col_id);

    int            line                () const;

    int            col                 () const;

    SyntaxVertices const& successors   () const;

    SyntaxVertex const* successor      () const;

    SyntaxVertex*  successor           ();

    SyntaxVertex const* successor      (size_type index) const;

    SyntaxVertex*  successor           (size_type index);

    void           add_successor       (SyntaxVertex* successor);

protected:

                   SyntaxVertex        ();

                   SyntaxVertex        (int line_nr,
                                        int col_id);

private:

    int            _line;

    int            _col;

    //! The next vertex/vertices to process.
    SyntaxVertices _successors;

};

} // namespace ranally
