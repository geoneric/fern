#pragma once


namespace fern {

//! A value is a property of an attribute's domain.
/*!
  \sa        .
*/
class Value
{

    friend class ValueTest;

public:

                   Value               (Value const&)=delete;

    Value&         operator=           (Value const&)=delete;

    virtual        ~Value              ();

protected:

                   Value               ();

private:

};

} // namespace fern
