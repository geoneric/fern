#pragma once


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Class
{

    friend class ClassTest;

public:

                   Class               ()=default;

                   Class               (Class const&)=delete;

    Class&         operator=           (Class const&)=delete;

                   Class               (Class&&)=delete;

    Class&         operator=           (Class&&)=delete;

                   ~Class              ()=default;

protected:

private:

};

} // namespace ranally
