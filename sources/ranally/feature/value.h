#pragma once


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Value
{

public:

protected:

                   Value               ()=default;

                   Value               (Value const&)=delete;

    Value&         operator=           (Value const&)=delete;

                   Value               (Value&&)=delete;

    Value&         operator=           (Value&&)=delete;

    virtual        ~Value              ();

private:

};

} // namespace ranally
