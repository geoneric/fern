#pragma once
#include "geoneric/operation/core/operation.h"


namespace geoneric {

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

    ResultType     result_type         (size_t index,
                        std::vector<ResultType> const& argument_types) const;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace geoneric
