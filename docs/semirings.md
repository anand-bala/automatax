Semirings
=========

Semirings
---------

A tuple, $\Ke = \left(K, \oplus, \otimes, \tilde{0}, \tilde{1}\right)$ is
a _semiring_ with the underlying set $K$ if

1. $\left(K, \oplus, \tilde{0}\right)$ is a commutative monoid with identity
   $\tilde{0}$;
2. $\left( K, \otimes, \tilde{1} \right)$ is a monoid with identity element $\tilde{1}$;
3. $\otimes$ distributes over $\oplus$; and
4. $\tilde{0}$ is an annihilator for $\otimes$ (for all $k \in K, k \otimes \tilde{0}
   = \tilde{0} \otimes k = \tilde{0}$).

In `automatix`, we define semirings as subclasses of the
[`AbstractSemiring`][automatix.semirings.AbstractSemiring] class, and overriding its
`ones`, `zeros`, `add`, `mul`, `sum`, and `prod` methods, which correspond to semiring
elements.


