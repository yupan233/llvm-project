//===-- Passes.td - HLFIR pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_OPENMP_PASSES
#define FORTRAN_DIALECT_OPENMP_PASSES

include "mlir/Pass/PassBase.td"

def LowerWorkshare : Pass<"lower-workshare"> {
  let summary = "Lower workshare construct";
}

#endif //FORTRAN_DIALECT_OPENMP_PASSES
