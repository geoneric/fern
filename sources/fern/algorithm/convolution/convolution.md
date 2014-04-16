Convolution  {#convolution}
===========

Theory
------
Computer Graphics, Principles and Practice:

The convolution of two signals \f$f(x)\f$ and \f$g(x)\f$, written as \f$f(x) * g(x)\f$, is a new signal \f$h(x)\f$ defined as follows. The value of \f$h(x)\f$ at each point is the integral of the product of \f$f(x)\f$ with the filter function \f$g(x)\f$ flipped about its vertical axis and shifted such that its origin is at that point. This corresponds to taking a weighted average of the neighborhood around each point of the signal \f$f(x)\f$, weighted by a flipped copy of filter \f$g(x)\f$ positioned at the point, and using it for the value of \f$h(x)\f$ at the point.

The size of the neighborhood is determined by the size of the domain over which the filter is nonzero. This is know as the filter's *support*, and a filter that is nonzero over a finite domain is said to have *finite support*.

Image Processing using 2D Convolution:

The size of the kernel, the numbers within it, and a single normalizer value define the operation that is applied to the image.

The kernel is applied to the image by placing the kernel over the image to be convolved and sliding it around to center it over every pixel in the original image. At each placement the numbers (pixel values) from the original image are multiplied by the kernel number that is currently aligned above it.

The sum of all these products is tabulated and divided by the kernel's normalizer. This result is placed into the new image at the position of the kernel's center. The kernel is translated to the next pixel position and the process repeats until all image pixels have been processed.

### Kernel ###
In image processing, a kernel is a small matrix that contains weights to be used to calculate new values based on a source image. For each pixel, the kernel is positioned with the center of the kernel on the current pixel. Surrounding pixels are weighted by the weights in the corresponding kernel cells, and the result is used as the value for the current pixel.

Synonyms: convolution kernel, convolution filer, convolution matrix, mask

Examples:

A 3x3 kernel that copies the input to the output:

    0 0 0
    0 1 0
    0 0 0

A 3x3 kernel that blurs the input image:

    1 1 1
    1 1 1
    1 1 1

Sharpen:

     0 -1  0
    -1  5 -1
     0 -1  0


### Neighborhood ###  {#neighborhood}
The neighborhood represents the shape of the non-zero values in the kernel. There are different neighborhood shapes:

- [Moore neighborhood](https://en.wikipedia.org/wiki/Moore_neighborhood)
- [(Extended) Von Neumann neighborhood](https://en.wikipedia.org/wiki/Von_Neumann_neighborhood)
- Square.
- Circle.


See also
--------
- [Wikipedia on convolution](https://en.wikipedia.org/wiki/Convolution)
- [Wikipedia on image processing kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)
- [Wikipedia on cellular automaton](https://en.wikipedia.org/wiki/Cellular_automata)
- [Computer Graphics, Principles and Practice](https://en.wikipedia.org/wiki/Computer_Graphics:_Principles_and_Practice)
- [Image Processing using 2D Convolution](http://williamson-labs.com/convolution-2d.htm)
