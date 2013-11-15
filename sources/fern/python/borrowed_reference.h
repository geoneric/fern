#pragma once
#include <Python.h>


namespace fern {
namespace python {

//! Class for holding a borrowed reference to a Python object.
/*!
  This class exists mainly for documentation purposes, so the user knows the
  layered object is borrowed, not owned.

  \sa        OwnedReference
*/
class BorrowedReference
{

    friend class BorrowedReferenceTest;

public:

                   BorrowedReference   ();

                   BorrowedReference   (PyObject* object);

    BorrowedReference& operator=       (PyObject* object);

                   BorrowedReference   (BorrowedReference&&)=delete;

    BorrowedReference& operator=       (BorrowedReference&&)=delete;

                   BorrowedReference   (BorrowedReference const& other);

    BorrowedReference& operator=       (BorrowedReference const&)=delete;

    //! Destruct instance.
    /*!
       The reference count of the layered object is not changed.
    */
                   ~BorrowedReference  ()=default;

                   operator bool       () const;

    PyObject**     operator&           ();

                   operator PyObject*  ();

    template<class T>
                   operator T*         ();

private:

    PyObject*        _object;

};


//! Return the layered Python object cast to \a T*
/*!
  \tparam    T Type to cast the layered Python object to.
  \return    Python object cast to T*.
  \warning   A Python object must be layered in the instance.
*/
template<class T>
inline BorrowedReference::operator T*()
{
    assert(_object);
    return (T*)_object;
}

} // namespace python
} // namespace fern
