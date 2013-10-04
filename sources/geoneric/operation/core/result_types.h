#pragma once
#include "geoneric/operation/core/result_type.h"


namespace geoneric {

class ResultTypes:
    public std::vector<ResultType>
{

public:

                   ResultTypes         ()=default;

                   ResultTypes         (std::initializer_list<ResultType> const& result_types);

                   ResultTypes         (size_t size);

                   ResultTypes         (ResultTypes&&)=default;

    ResultTypes&   operator=           (ResultTypes&&)=default;

                   ResultTypes         (ResultTypes const&)=default;

    ResultTypes&   operator=           (ResultTypes const&)=default;

                   ~ResultTypes        ()=default;

    bool           is_satisfied_by     (ResultTypes const& result_types) const;

    size_t         id_of_satisfying_type(
                                        ResultTypes const& result_types) const;

    bool           fixed               () const;

private:

    bool           is_satisfied_by     (ResultType const& result_type) const;

};


std::ostream&      operator<<          (std::ostream& stream,
                                        ResultTypes const& result_types);

} // namespace geoneric
