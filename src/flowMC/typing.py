"""Convenience type aliases for jaxtyping scalar annotations."""

from jaxtyping import Array, ArrayLike, Complex, Float, Int, Real

type FloatScalar = Float[Array, ""]
type IntScalar = Int[Array, ""]
type ComplexScalar = Complex[Array, ""]
type FloatLike = float | Float[Array, ""]
type RealScalarLike = Real[ArrayLike, ""]
