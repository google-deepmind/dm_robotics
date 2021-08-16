# Parametric object class

Warning: within the context of this codebase an object is a physical object, not
an object in the software engineering sense. Therefore, we will often refer to
the object shape and the object texture.

This codebase contains a class to manipulate `ParametricObject`. A parametric
object is the combination of two `ParametricProperties`, one for the shape and
one for the texture of the object itself. A parametric property is a collection
of strings (the name of the property, e.g. `width`), numbers (the value of that
property) and bounds for these properties. Optionally, the value of properties
can be associated with a unit of measure (e.g. meters for lengths and radians
for angles). The classes allow the user to create a parametric object and
manipulate it as needed.

Examples of the use of this class are the [RGB-objects][RgbDocumentation].

Links:

-   [OnShape API][OnShapeAPI],

-   [RGB-objects][RgbDocumentation],

-   [Manipulating the RGB-objects &#128721;&#129001;&#128311;][RgbDocumentation].


<!-- Hyperlinks  -->
[OnShapeAPI]: https://onshape-public.github.io/docs/

[RgbDocumentation]: ./



