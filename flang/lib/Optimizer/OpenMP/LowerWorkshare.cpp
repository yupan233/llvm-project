//===- LowerWorkshare.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lower omp workshare construct.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"

#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKSHARE
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workshare"

using namespace mlir;

namespace flangomp {
bool shouldUseWorkshareLowering(Operation *op) {
  auto workshare = dyn_cast<omp::WorkshareOp>(op->getParentOp());
  if (!workshare)
    return false;
  return workshare->getParentOfType<omp::ParallelOp>();
}
} // namespace flangomp

namespace {

struct SingleRegion {
  Block::iterator begin, end;
};

static bool isSupportedByFirAlloca(Type ty) {
  return !isa<fir::ReferenceType>(ty);
}

static bool isSafeToParallelize(Operation *op) {
  if (isa<fir::DeclareOp>(op))
    return true;

  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) {
    return false;
  }
  interface.getEffects(effects);
  if (effects.empty())
    return true;

  return false;
}

/// Lowers workshare to a sequence of single-thread regions and parallel loops
///
/// For example:
///
/// omp.workshare {
///   %a = fir.allocmem
///   omp.wsloop {}
///   fir.call Assign %b %a
///   fir.freemem %a
/// }
///
/// becomes
///
/// omp.single {
///   %a = fir.allocmem
///   fir.store %a %tmp
/// }
/// %a_reloaded = fir.load %tmp
/// omp.wsloop {}
/// omp.single {
///   fir.call Assign %b %a_reloaded
///   fir.freemem %a_reloaded
/// }
///
/// Note that we allocate temporary memory for values in omp.single's which need
/// to be accessed in all threads in the closest omp.parallel
///
/// TODO currently we need to be able to access the encompassing omp.parallel so
/// that we can allocate temporaries accessible by all threads outside of it.
/// In case we do not find it, we fall back to converting the omp.workshare to
/// omp.single.
/// To better handle this we should probably enable yielding values out of an
/// omp.single which will be supported by the omp runtime.
void lowerWorkshare(mlir::omp::WorkshareOp wsOp) {
  assert(wsOp.getRegion().getBlocks().size() == 1);

  Location loc = wsOp->getLoc();

  omp::ParallelOp parallelOp = wsOp->getParentOfType<omp::ParallelOp>();
  if (!parallelOp) {
    wsOp.emitWarning("cannot handle workshare, converting to single");
    Operation *terminator = wsOp.getRegion().front().getTerminator();
    wsOp->getBlock()->getOperations().splice(
        wsOp->getIterator(), wsOp.getRegion().front().getOperations());
    terminator->erase();
    return;
  }

  OpBuilder allocBuilder(parallelOp);
  OpBuilder rootBuilder(wsOp);
  IRMapping rootMapping;

  omp::SingleOp singleOp = nullptr;

  auto mapReloadedValue = [&](Value v, OpBuilder singleBuilder,
                              IRMapping singleMapping) {
    if (auto reloaded = rootMapping.lookupOrNull(v))
      return;
    Type llvmPtrTy = LLVM::LLVMPointerType::get(allocBuilder.getContext());
    Type ty = v.getType();
    Value alloc, reloaded;
    if (isSupportedByFirAlloca(ty)) {
      alloc = allocBuilder.create<fir::AllocaOp>(loc, ty);
      singleBuilder.create<fir::StoreOp>(loc, singleMapping.lookup(v), alloc);
      reloaded = rootBuilder.create<fir::LoadOp>(loc, ty, alloc);
    } else {
      auto one = allocBuilder.create<LLVM::ConstantOp>(
          loc, allocBuilder.getI32Type(), 1);
      alloc =
          allocBuilder.create<LLVM::AllocaOp>(loc, llvmPtrTy, llvmPtrTy, one);
      Value toStore = singleBuilder
                          .create<UnrealizedConversionCastOp>(
                              loc, llvmPtrTy, singleMapping.lookup(v))
                          .getResult(0);
      singleBuilder.create<LLVM::StoreOp>(loc, toStore, alloc);
      reloaded = rootBuilder.create<LLVM::LoadOp>(loc, llvmPtrTy, alloc);
      reloaded =
          rootBuilder.create<UnrealizedConversionCastOp>(loc, ty, reloaded)
              .getResult(0);
    }
    rootMapping.map(v, reloaded);
  };

  auto moveToSingle = [&](SingleRegion sr, OpBuilder singleBuilder) {
    IRMapping singleMapping = rootMapping;

    for (Operation &op : llvm::make_range(sr.begin, sr.end)) {
      singleBuilder.clone(op, singleMapping);
      if (isSafeToParallelize(&op)) {
        rootBuilder.clone(op, rootMapping);
      } else {
        // Prepare reloaded values for results of operations that cannot be
        // safely parallelized and which are used after the region `sr`
        for (auto res : op.getResults()) {
          for (auto &use : res.getUses()) {
            Operation *user = use.getOwner();
            while (user->getParentOp() != wsOp)
              user = user->getParentOp();
            if (!user->isBeforeInBlock(&*sr.end)) {
              // We need to reload
              mapReloadedValue(use.get(), singleBuilder, singleMapping);
            }
          }
        }
      }
    }
    singleBuilder.create<omp::TerminatorOp>(loc);
  };

  Block *wsBlock = &wsOp.getRegion().front();
  assert(wsBlock->getTerminator()->getNumOperands() == 0);
  Operation *terminator = wsBlock->getTerminator();

  SmallVector<std::variant<SingleRegion, omp::WsloopOp>> regions;

  auto it = wsBlock->begin();
  auto getSingleRegion = [&]() {
    if (&*it == terminator)
      return false;
    if (auto pop = dyn_cast<omp::WsloopOp>(&*it)) {
      regions.push_back(pop);
      it++;
      return true;
    }
    SingleRegion sr;
    sr.begin = it;
    while (&*it != terminator && !isa<omp::WsloopOp>(&*it))
      it++;
    sr.end = it;
    assert(sr.begin != sr.end);
    regions.push_back(sr);
    return true;
  };
  while (getSingleRegion())
    ;

  for (auto [i, loopOrSingle] : llvm::enumerate(regions)) {
    bool isLast = i + 1 == regions.size();
    if (std::holds_alternative<SingleRegion>(loopOrSingle)) {
      omp::SingleOperands singleOperands;
      if (isLast)
        singleOperands.nowait = rootBuilder.getUnitAttr();
      singleOp = rootBuilder.create<omp::SingleOp>(loc, singleOperands);
      OpBuilder singleBuilder(singleOp);
      singleBuilder.createBlock(&singleOp.getRegion());
      moveToSingle(std::get<SingleRegion>(loopOrSingle), singleBuilder);
    } else {
      rootBuilder.clone(*std::get<omp::WsloopOp>(loopOrSingle), rootMapping);
      if (!isLast)
        rootBuilder.create<omp::BarrierOp>(loc);
    }
  }

  if (!wsOp.getNowait())
    rootBuilder.create<omp::BarrierOp>(loc);

  wsOp->erase();

  return;
}

class LowerWorksharePass
    : public flangomp::impl::LowerWorkshareBase<LowerWorksharePass> {
public:
  void runOnOperation() override {
    SmallPtrSet<Operation *, 8> parents;
    getOperation()->walk([&](mlir::omp::WorkshareOp wsOp) {
      Operation *isolatedParent =
          wsOp->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
      parents.insert(isolatedParent);

      lowerWorkshare(wsOp);
    });

    // Do folding
    for (Operation *isolatedParent : parents) {
      RewritePatternSet patterns(&getContext());
      GreedyRewriteConfig config;
      // prevent the pattern driver form merging blocks
      config.enableRegionSimplification =
          mlir::GreedySimplifyRegionLevel::Disabled;
      if (failed(applyPatternsAndFoldGreedily(isolatedParent,
                                              std::move(patterns), config))) {
        emitError(isolatedParent->getLoc(), "error in lower workshare\n");
        signalPassFailure();
      }
    }
  }
};
} // namespace
