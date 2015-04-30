// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/expression_type.h"


namespace fern {
namespace language {

class ExpressionTypes:
    private std::vector<ExpressionType>
{

public:

                   ExpressionTypes     ()=default;

                   ExpressionTypes     (
              std::initializer_list<ExpressionType> const& expression_types);

                   ExpressionTypes     (size_t size);

                   ExpressionTypes     (ExpressionTypes&&)=default;

    ExpressionTypes& operator=         (ExpressionTypes&&)=default;

                   ExpressionTypes     (ExpressionTypes const&)=default;

    ExpressionTypes& operator=         (ExpressionTypes const&)=default;

                   ~ExpressionTypes    ()=default;

    bool           is_empty            () const;

    size_t         size                () const;

    void           add                 (ExpressionType const& expression_type);

    bool           is_satisfied_by     (
                             ExpressionTypes const& expression_types) const;

    size_t         id_of_satisfying_type(
                             ExpressionTypes const& expression_types) const;

    bool           fixed               () const;

    ExpressionType const& operator[]   (size_t index) const;

    ExpressionType& operator[]         (size_t index);

    const_iterator begin               () const;

    const_iterator end                 () const;

private:

    bool           is_satisfied_by     (
                                  ExpressionType const& expression_type) const;

};


std::ostream&      operator<<          (std::ostream& stream,
                                  ExpressionTypes const& expression_types);

} // namespace language
} // namespace fern
