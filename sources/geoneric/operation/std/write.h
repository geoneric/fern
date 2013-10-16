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

    ExpressionType expression_type     (size_t index,
                   std::vector<ExpressionType> const& argument_types) const;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace geoneric
