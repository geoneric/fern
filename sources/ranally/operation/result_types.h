#pragma once
#include "ranally/operation/result_type.h"


namespace ranally {

class ResultTypes:
    public std::vector<ResultType>
{

    friend class ResultTypeTest;

public:

                   ResultTypes         ()=default;

                   ResultTypes         (ResultTypes&&)=default;

    ResultTypes&   operator=           (ResultTypes&&)=default;

                   ResultTypes         (ResultTypes const&)=default;

    ResultTypes&   operator=           (ResultTypes const&)=default;

                   ~ResultTypes        ()=default;

    bool           fixed               () const;

private:

};

} // namespace ranally
