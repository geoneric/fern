#pragma once
#include <Python.h>


namespace ranally {
namespace python {

//! Class for holding an owned reference to a Python object.
/*!
  This class exists mainly for documentation purposes, so the user knows the
  layered object is owned, not borrowed. Also, reference counts of the owned
  references are managed.

  \sa        BorrowedReference
*/
class OwnedReference
{

    friend class OwnedReferenceTest;

public:

                   OwnedReference      ();

                   OwnedReference      (PyObject* object);

    OwnedReference& operator=          (PyObject* object);

                   OwnedReference      (OwnedReference&&)=delete;

    OwnedReference& operator=          (OwnedReference&&)=delete;

                   OwnedReference      (OwnedReference const&)=delete;

    OwnedReference& operator=          (OwnedReference const&)=delete;

                   ~OwnedReference     ();

                   operator bool       () const;

    PyObject**     operator&           ();

                   operator PyObject*  ();

    void           incRef              ();

private:

    PyObject*        _object;

    void           decRef              ();

};

} // namespace python
} // namespace ranally
