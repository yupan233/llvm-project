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
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/iterator_range.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/OpenMP/OpenMPClauseOperands.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
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

static bool mustParallelizeOp(Operation *op) {
  return op
      ->walk(
          [](omp::WorkshareLoopWrapperOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

static bool isSafeToParallelize(Operation *op) {
  return isa<fir::DeclareOp>(op) || isPure(op);
}

static void parallelizeRegion(Region &sourceRegion, Region &targetRegion,
                              IRMapping &rootMapping, Location loc) {
  Operation *parentOp = sourceRegion.getParentOp();
  OpBuilder rootBuilder(sourceRegion.getContext());

  // TODO need to copyprivate the alloca's
  auto mapReloadedValue = [&](Value v, OpBuilder singleBuilder,
                              IRMapping singleMapping) {
    OpBuilder allocaBuilder(&targetRegion.front().front());
    if (auto reloaded = rootMapping.lookupOrNull(v))
      return;
    Type llvmPtrTy = LLVM::LLVMPointerType::get(allocaBuilder.getContext());
    Type ty = v.getType();
    Value alloc, reloaded;
    if (isSupportedByFirAlloca(ty)) {
      alloc = allocaBuilder.create<fir::AllocaOp>(loc, ty);
      singleBuilder.create<fir::StoreOp>(loc, singleMapping.lookup(v), alloc);
      reloaded = rootBuilder.create<fir::LoadOp>(loc, ty, alloc);
    } else {
      auto one = allocaBuilder.create<LLVM::ConstantOp>(
          loc, allocaBuilder.getI32Type(), 1);
      alloc =
          allocaBuilder.create<LLVM::AllocaOp>(loc, llvmPtrTy, llvmPtrTy, one);
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
            while (user->getParentOp() != parentOp)
              user = user->getParentOp();
            if (!(user->isBeforeInBlock(&*sr.end) &&
                  sr.begin->isBeforeInBlock(user))) {
              // We need to reload
              mapReloadedValue(use.get(), singleBuilder, singleMapping);
            }
          }
        }
      }
    }
    singleBuilder.create<omp::TerminatorOp>(loc);
  };

  // TODO Need to handle these (clone them) in dominator tree order
  for (Block &block : sourceRegion) {
    rootBuilder.createBlock(
        &targetRegion, {}, block.getArgumentTypes(),
        llvm::map_to_vector(block.getArguments(),
                            [](BlockArgument arg) { return arg.getLoc(); }));
    Operation *terminator = block.getTerminator();

    SmallVector<std::variant<SingleRegion, Operation *>> regions;

    auto it = block.begin();
    auto getOneRegion = [&]() {
      if (&*it == terminator)
        return false;
      if (mustParallelizeOp(&*it)) {
        regions.push_back(&*it);
        it++;
        return true;
      }
      SingleRegion sr;
      sr.begin = it;
      while (&*it != terminator && !mustParallelizeOp(&*it))
        it++;
      sr.end = it;
      assert(sr.begin != sr.end);
      regions.push_back(sr);
      return true;
    };
    while (getOneRegion())
      ;

    for (auto [i, opOrSingle] : llvm::enumerate(regions)) {
      bool isLast = i + 1 == regions.size();
      if (std::holds_alternative<SingleRegion>(opOrSingle)) {
        omp::SingleOperands singleOperands;
        if (isLast)
          singleOperands.nowait = rootBuilder.getUnitAttr();
        omp::SingleOp singleOp =
            rootBuilder.create<omp::SingleOp>(loc, singleOperands);
        OpBuilder singleBuilder(singleOp);
        singleBuilder.createBlock(&singleOp.getRegion());
        moveToSingle(std::get<SingleRegion>(opOrSingle), singleBuilder);
      } else {
        auto op = std::get<Operation *>(opOrSingle);
        if (auto wslw = dyn_cast<omp::WorkshareLoopWrapperOp>(op)) {
          omp::WsloopOperands wsloopOperands;
          if (isLast)
            wsloopOperands.nowait = rootBuilder.getUnitAttr();
          auto wsloop =
              rootBuilder.create<mlir::omp::WsloopOp>(loc, wsloopOperands);
          auto clonedWslw = cast<omp::WorkshareLoopWrapperOp>(
              rootBuilder.clone(*wslw, rootMapping));
          wsloop.getRegion().takeBody(clonedWslw.getRegion());
          clonedWslw->erase();
        } else {
          assert(mustParallelizeOp(op));
          Operation *cloned = rootBuilder.cloneWithoutRegions(*op, rootMapping);
          for (auto [region, clonedRegion] :
               llvm::zip(op->getRegions(), cloned->getRegions()))
            parallelizeRegion(region, clonedRegion, rootMapping, loc);
        }
      }
    }

    rootBuilder.clone(*block.getTerminator(), rootMapping);
  }
}

/// Lowers workshare to a sequence of single-thread regions and parallel loops
///
/// For example:
///
/// omp.workshare {
///   %a = fir.allocmem
///   omp.workshare_loop_wrapper {}
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
/// omp.workshare_loop_wrapper {}
/// omp.single {
///   fir.call Assign %b %a_reloaded
///   fir.freemem %a_reloaded
/// }
///
/// Note that we allocate temporary memory for values in omp.single's which need
/// to be accessed in all threads in the closest omp.parallel
void lowerWorkshare(mlir::omp::WorkshareOp wsOp) {
  Location loc = wsOp->getLoc();
  IRMapping rootMapping;

  OpBuilder rootBuilder(wsOp);

  // TODO We need something like an scf;execute here, but that is not registered
  // so using fir.if for now but it looks like it does not support multiple
  // blocks so it doesnt work for multi block case...
  auto ifOp = rootBuilder.create<fir::IfOp>(
      loc, rootBuilder.create<arith::ConstantIntOp>(loc, 1, 1), false);
  ifOp.getThenRegion().front().erase();

  parallelizeRegion(wsOp.getRegion(), ifOp.getThenRegion(), rootMapping, loc);

  Operation *terminatorOp = ifOp.getThenRegion().back().getTerminator();
  assert(isa<omp::TerminatorOp>(terminatorOp));
  OpBuilder termBuilder(terminatorOp);

  if (!wsOp.getNowait())
    termBuilder.create<omp::BarrierOp>(loc);

  termBuilder.create<fir::ResultOp>(loc, ValueRange());

  terminatorOp->erase();
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
