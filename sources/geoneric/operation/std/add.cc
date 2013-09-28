#include "geoneric/operation/std/add.h"
#include "geoneric/operation/core/attribute_argument.h"


namespace geoneric {

Add::Add()

    : Operation("add",
          "Add the two argument values and return the result.",
          {
              Parameter("First input argument",
                  "First argument to add.",
                  DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                  ValueTypes::NUMBER),
              Parameter("Second input argument",
                  "Second argument to add.",
                  DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                  ValueTypes::NUMBER)
          },
          {
              Result("Result value",
                  "Input arguments, added together.",
                  DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                  ValueTypes::NUMBER)
          }
      )

{
}


std::vector<std::shared_ptr<Argument>> Add::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 2);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);
    assert(arguments[1]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    AttributeArgument const& attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[0]));
    assert(attribute_argument.data_type() == DataType::DT_CONSTANT);
    assert(attribute_argument.value_type() == ValueType::VT_INT64);

    return std::vector<std::shared_ptr<Argument>>();
}

} // namespace geoneric
