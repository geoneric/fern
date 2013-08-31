#pragma once
#include "geoneric/operation/core/result_type.h"


namespace geoneric {

class ResultTypes:
    public std::vector<ResultType>
{

    friend class ResultTypeTest;

public:

                   ResultTypes         ()=default;

                   ResultTypes         (size_t size);

                   ResultTypes         (ResultTypes&&)=default;

    ResultTypes&   operator=           (ResultTypes&&)=default;

                   ResultTypes         (ResultTypes const&)=default;

    ResultTypes&   operator=           (ResultTypes const&)=default;

                   ~ResultTypes        ()=default;

    bool           fixed               () const;

private:

};

} // namespace geoneric
