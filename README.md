# VectorBijectors.jl

[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliabayes.org/VectorBijectors.jl/)

VectorBijectors.jl converts samples from distributions to and from **vectors**.

It assumes that there are three forms of samples from a distribution `d` that we are interested in:

 1. **The original form**, which is what `rand(d)` returns.

 2. **A vectorised form**, which is a vector that contains a flattened version of the original form.

 3. **A linked vectorised form**, which is a vector in which:

      + each element is independent; and
      + each element is unconstrained (can take any value in ℝ).

and provides functionality to convert between these three forms.

On top of defining a clearer interface than Bijectors.jl (from which much code is lifted), VectorBijectors also has the aim of being:

 1. Usable on GPUs, particularly via Reactant.jl.

 2. More performant and compatible with modern automatic differentiation.

Much of this is in fact contingent on the underlying distributions.
To this end, VectorBijectors is also intentionally decoupled from Distributions.jl: the pre-existing functionality for Distributions.jl is provided in an extension.
This allows other distribution providers to use the functionality in this library as well.
