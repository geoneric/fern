#pragma once
#include "fern/operation/core/operation.h"


namespace fern {

class Write:
    public Operation
{

public:

                   Write               ();

                   ~Write              ()=default;

                   Write               (Write&& other)=delete;

    Write&         operator=           (Write&& other)=delete;

                   Write               (Write const& other)=delete;

    Write&         operator=           (Write const& other)=delete;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace fern
