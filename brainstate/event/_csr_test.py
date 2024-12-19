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
# -*- coding: utf-8 -*-


import unittest

import brainunit as u

import brainstate as bst


class TestCSR(unittest.TestCase):
    def test_event_homo_bool(self):
        for dat in [1., 2., 3.]:
            mask = (bst.random.rand(10, 20) < 0.1).astype(float) * dat
            csr = u.sparse.CSR.fromdense(mask)
            csr = bst.event.CSR((dat, csr.indices, csr.indptr), shape=mask.shape)

            v = bst.random.rand(20) < 0.5
            self.assertTrue(
                u.math.allclose(
                    mask.astype(float) @ v.astype(float),
                    csr @ v
                )
            )

            v = bst.random.rand(10) < 0.5
            self.assertTrue(
                u.math.allclose(
                    v.astype(float) @ mask.astype(float),
                    v @ csr
                )
            )

    def test_event_homo_heter(self):
        mat = bst.random.rand(10, 20)
        mask = (bst.random.rand(10, 20) < 0.1) * mat
        csr = u.sparse.CSR.fromdense(mask)
        csr = bst.event.CSR((csr.data, csr.indices, csr.indptr), shape=mask.shape)

        v = bst.random.rand(20) < 0.5
        self.assertTrue(
            u.math.allclose(
                mask.astype(float) @ v.astype(float),
                csr @ v
            )
        )

        v = bst.random.rand(10) < 0.5
        self.assertTrue(
            u.math.allclose(
                v.astype(float) @ mask.astype(float),
                v @ csr
            )
        )

    def test_event_heter_float_as_bool(self):
        mat = bst.random.rand(10, 20)
        mask = (mat < 0.1).astype(float) * mat
        csr = u.sparse.CSR.fromdense(mask)
        csr = bst.event.CSR((csr.data, csr.indices, csr.indptr), shape=mask.shape)

        v = (bst.random.rand(20) < 0.5).astype(float)
        self.assertTrue(
            u.math.allclose(
                mask.astype(float) @ v.astype(float),
                csr @ v
            )
        )

        v = (bst.random.rand(10) < 0.5).astype(float)
        self.assertTrue(
            u.math.allclose(
                v.astype(float) @ mask.astype(float),
                v @ csr
            )
        )
