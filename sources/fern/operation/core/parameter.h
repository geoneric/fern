#pragma once
#include "fern/core/string.h"
#include "fern/operation/core/expression_types.h"


namespace fern {

//! A parameter is a description of an operation's argument.
/*!
  A parameter is defined by its name, description, and the types of data and
  values it is satisfied with.

  \sa        .
*/
class Parameter
{

    friend class ParameterTest;

public:

                   Parameter           (String const& name,
                                        String const& description,
                                        DataTypes data_types,
                                        ValueTypes value_types);

                   ~Parameter          ();

                   Parameter           (Parameter&& other);

    Parameter&     operator=           (Parameter&& other);

                   Parameter           (Parameter const& other);

    Parameter&     operator=           (Parameter const& other);

    String const&  name                () const;

    String const&  description         () const;

    ExpressionTypes expression_types   () const;

private:

    String         _name;

    String         _description;

    ExpressionTypes _expression_types;

};

} // namespace fern