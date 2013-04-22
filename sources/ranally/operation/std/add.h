#pragma once
#include "ranally/operation/core/operation.h"


namespace ranally {

class Add:
    public Operation
{

public:

                   Add                 ();

                   ~Add                ()=default;

                   Add                 (Add&& other)=delete;

    Add&           operator=           (Add&& other)=delete;

                   Add                 (Add const& other)=delete;

    Add&           operator=           (Add const& other)=delete;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace ranally
