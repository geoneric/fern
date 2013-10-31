#pragma once
#include "geoneric/operation/core/operation.h"


namespace geoneric {

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

} // namespace geoneric
