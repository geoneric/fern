#pragma once
#include "geoneric/operation/core/operation.h"


namespace geoneric {

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

} // namespace geoneric
