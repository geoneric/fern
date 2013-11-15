#include "fern/operation/std/add.h"
#include "fern/operation/core/attribute_argument.h"
#include "fern/operation/core/expression_type_calculation.h"


namespace fern {

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
                  ExpressionType(DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                      ValueTypes::NUMBER))
          }
      )

{
}


ExpressionType Add::expression_type(
    size_t index,
    std::vector<ExpressionType> const& argument_types) const
{
    return standard_expression_type(*this, index, argument_types);
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

} // namespace fern
