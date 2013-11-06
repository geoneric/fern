#pragma once


namespace geoneric {

class Module
{

public:

    virtual void   run                 ();

protected:

                   Module              ()=default;

    virtual        ~Module             ()=default;

                   Module              (Module&&)=delete;

    Module&        operator=           (Module&&)=delete;

                   Module              (Module const&)=delete;

    Module&        operator=           (Module const&)=delete;

private:

};

} // namespace geoneric
