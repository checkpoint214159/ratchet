"""Hand-written Triton kernels.

They live in real source files because Triton's JIT reads the decorated function's SOURCE
to compile it -- a `@triton.jit` function defined in stdin or a heredoc raises
`OSError: could not get source code`.
"""
