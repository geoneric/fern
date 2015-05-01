// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include <vector>
#include <loki/Visitor.h>


namespace fern {
namespace language {

class ExpressionVertex;
class StatementVertex;

using ExpressionVertices = std::vector<std::shared_ptr<ExpressionVertex>>;
using StatementVertices = std::vector<std::shared_ptr<StatementVertex>>;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AstVertex:
    public Loki::BaseVisitable<>
{

private:

    using AstVertices = std::vector<AstVertex*>;

public:

    LOKI_DEFINE_VISITABLE()

    using size_type = AstVertices::size_type;

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

} // namespace language
} // namespace fern
