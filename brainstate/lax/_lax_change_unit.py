# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

__all__ = [
    # math funcs change unit (unary)
    'cumprod', 'sqrt', 'rsqrt', 'cbrt', 'square', 'reciprocal',

    # math funcs change unit (binary)
    'conv', 'conv_transpose', 'div', 'dot', 'dot_general',
    'pow', 'integer_pow', 'mul', 'div', 'rem',

    # math funcs change unit (other)
    'batch_matmul',

    # linear algebra
    'cholesky', 'eig', 'eigh', 'hessenberg', 'lu',
    'householder_product', 'qdwh', 'qr', 'schur', 'svd', 'triangular_solve',
    'tridiagonal', 'tridiagonal_solve',

    # fft
    'fft',
]

# math funcs change unit (unary)
def cumprod(x): pass
def sqrt(x): pass
def rsqrt(x): pass
def cbrt(x): pass
def square(x): pass
def reciprocal(x): pass

# math funcs change unit (binary)
def conv(x, y): pass
def conv_transpose(x, y): pass
def div(x, y): pass
def dot(x, y): pass
def dot_general(x, y): pass
def pow(x, y): pass
def integer_pow(x, y): pass
def mul(x, y): pass
def rem(x, y): pass

# math funcs change unit (other)
def batch_matmul(x, y): pass

# linear algebra
def cholesky(x): pass
def eig(x): pass
def eigh(x): pass
def hessenberg(x): pass
def lu(x): pass
def householder_product(x): pass
def qdwh(x): pass
def qr(x): pass
def schur(x): pass
def svd(x): pass
def triangular_solve(x): pass
def tridiagonal(x): pass
def tridiagonal_solve(x): pass

# fft
def fft(x): pass