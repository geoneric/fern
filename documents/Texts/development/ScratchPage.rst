************
Scratch page
************

GPGPU
=====
Strategy for using the GPU for computations:

Given an expression tree:

* Determine how much memory we can use on the GPU.
* Determine how much memory we need on the GPU.
* Determine max block size that will result in memory GPU completely filled.
* Allocate memory for input attribute values and output attribute values in the GPU memory.
* For each block:

  * Copy blocks with input attribute values to the GPU memory.
  * Perform all operations on the GPU.
  * Copy blocks with result attribute values from the GPU to the host.

Components.txt

.. literalinclude:: Components.txt

Brainstorm.txt

.. literalinclude:: Brainstorm.txt

Introduction.txt

.. literalinclude:: Introduction.txt


