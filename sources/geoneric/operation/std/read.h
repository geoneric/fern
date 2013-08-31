#pragma once
#include "ranally/operation/core/operation.h"


namespace ranally {

class Read:
    public Operation
{

public:

                   Read                ();

                   ~Read               ()=default;

                   Read                (Read&& other)=delete;

    Read&          operator=           (Read&& other)=delete;

                   Read                (Read const& other)=delete;

    Read&          operator=           (Read const& other)=delete;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace ranally
