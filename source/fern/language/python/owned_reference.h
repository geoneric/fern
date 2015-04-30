// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <Python.h>


namespace fern {
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

    void           inc_ref             ();

private:

    PyObject*        _object;

    void           dec_ref             ();

};

} // namespace python
} // namespace fern
