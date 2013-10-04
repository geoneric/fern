#include "geoneric/operation/std/int32.h"
#include "geoneric/operation/core/attribute_argument.h"


namespace geoneric {

Int32::Int32()

    : Operation("int32",
          "Cast the argument to int32 and return the result.",
          {
              Parameter("Input argument",
                  "Argument to cast.",
                  DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                  ValueTypes::NUMBER)
          },
          {
              Result("Result value",
                  "Input argument as an int32.",
                  ResultType(DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                      ValueTypes::INT32))
          }
      )

{
}


std::vector<std::shared_ptr<Argument>> Int32::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 1);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    assert(false);
}

} // namespace geoneric
