#pragma once
#include "ranally/operation/core/operation.h"


namespace ranally {

class Abs:
    public Operation
{

public:

                   Abs                 ();

                   ~Abs                ()=default;

                   Abs                 (Abs&& other)=delete;

    Abs&           operator=           (Abs&& other)=delete;

                   Abs                 (Abs const& other)=delete;

    Abs&           operator=           (Abs const& other)=delete;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace ranally
