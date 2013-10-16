#pragma once
#include "geoneric/operation/core/expression_type.h"


namespace geoneric {

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

} // namespace geoneric
