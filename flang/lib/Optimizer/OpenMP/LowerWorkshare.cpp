//===- LowerWorkshare.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lower omp workshare construct.
//===----------------------------------------------------------------------===//

#include <flang/Optimizer/Builder/FIRBuilder.h>
#include <flang/Optimizer/Dialect/FIROps.h>
#include <flang/Optimizer/Dialect/FIRType.h>
#include <flang/Optimizer/HLFIR/HLFIROps.h>
#include <flang/Optimizer/OpenMP/Passes.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKSHARE
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workshare"

using namespace mlir;

namespace flangomp {
bool shouldUseWorkshareLowering(Operation *op) {
  // TODO this is insufficient, as we could have
  // omp.parallel {
  //   omp.workshare {
  //     omp.parallel {
  //       hlfir.elemental {}
  //
  // Then this hlfir.elemental shall _not_ use the lowering for workshare
  //
  // Standard says:
  //   For a parallel construct, the construct is a unit of work with respect to
  //   the workshare construct. The statements contained in the parallel
  //   construct are executed by a new thread team.
  //
  // TODO similarly for single, critical, etc. Need to think through the
  // patterns and implement this function.
  //
  return op->getParentOfType<omp::WorkshareOp>();
}
} // namespace flangomp

namespace {

struct SingleRegion {
  Block::iterator begin, end;
};

static bool mustParallelizeOp(Operation *op) {
  // TODO as in shouldUseWorkshareLowering we be careful not to pick up
  // workshare_loop_wrapper in nested omp.parallel ops
  //
  // e.g.
  //
  // omp.parallel {
  //   omp.workshare {
  //     omp.parallel {
  //       omp.workshare {
  //         omp.workshare_loop_wrapper {}
  return op
      ->walk(
          [](omp::WorkshareLoopWrapperOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

static bool isSafeToParallelize(Operation *op) {
  return isa<hlfir::DeclareOp>(op) || isa<fir::DeclareOp>(op) ||
         isMemoryEffectFree(op);
}

static mlir::func::FuncOp createCopyFunc(mlir::Location loc, mlir::Type varType,
                                         fir::FirOpBuilder builder) {
  mlir::ModuleOp module = builder.getModule();
  auto rt = cast<fir::ReferenceType>(varType);
  mlir::Type eleTy = rt.getEleTy();
  std::string copyFuncName =
      fir::getTypeAsString(eleTy, builder.getKindMap(), "_workshare_copy");

  if (auto decl = module.lookupSymbol<mlir::func::FuncOp>(copyFuncName))
    return decl;
  // create function
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::OpBuilder modBuilder(module.getBodyRegion());
  llvm::SmallVector<mlir::Type> argsTy = {varType, varType};
  auto funcType = mlir::FunctionType::get(builder.getContext(), argsTy, {});
  mlir::func::FuncOp funcOp =
      modBuilder.create<mlir::func::FuncOp>(loc, copyFuncName, funcType);
  funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  builder.createBlock(&funcOp.getRegion(), funcOp.getRegion().end(), argsTy,
                      {loc, loc});
  builder.setInsertionPointToStart(&funcOp.getRegion().back());

  Value loaded = builder.create<fir::LoadOp>(loc, funcOp.getArgument(0));
  builder.create<fir::StoreOp>(loc, loaded, funcOp.getArgument(1));

  builder.create<mlir::func::ReturnOp>(loc);
  return funcOp;
}

static bool isUserOutsideSR(Operation *user, Operation *parentOp,
                            SingleRegion sr) {
  while (user->getParentOp() != parentOp)
    user = user->getParentOp();
  return sr.begin->getBlock() != user->getBlock() ||
         !(user->isBeforeInBlock(&*sr.end) && sr.begin->isBeforeInBlock(user));
}

static bool isTransitivelyUsedOutside(Value v, SingleRegion sr) {
  Block *srBlock = sr.begin->getBlock();
  Operation *parentOp = srBlock->getParentOp();

  for (auto &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (isUserOutsideSR(user, parentOp, sr))
      return true;

    // Results of nested users cannot be used outside of the SR
    if (user->getBlock() != srBlock)
      continue;

    // A non-safe to parallelize operation will be handled separately
    if (!isSafeToParallelize(user))
      continue;

    for (auto res : user->getResults())
      if (isTransitivelyUsedOutside(res, sr))
        return true;
  }
  return false;
}

/// We clone pure operations in both the parallel and single blocks. this
/// functions cleans them up if they end up with no uses
static void cleanupBlock(Block *block) {
  for (Operation &op : llvm::make_early_inc_range(*block))
    if (isOpTriviallyDead(&op))
      op.erase();
}

static void parallelizeRegion(Region &sourceRegion, Region &targetRegion,
                              IRMapping &rootMapping, Location loc) {
  OpBuilder rootBuilder(sourceRegion.getContext());
  ModuleOp m = sourceRegion.getParentOfType<ModuleOp>();
  OpBuilder copyFuncBuilder(m.getBodyRegion());
  fir::FirOpBuilder firCopyFuncBuilder(copyFuncBuilder, m);

  auto mapReloadedValue =
      [&](Value v, OpBuilder allocaBuilder, OpBuilder singleBuilder,
          OpBuilder parallelBuilder, IRMapping singleMapping) -> Value {
    if (auto reloaded = rootMapping.lookupOrNull(v))
      return nullptr;
    Type ty = v.getType();
    Value alloc = allocaBuilder.create<fir::AllocaOp>(loc, ty);
    singleBuilder.create<fir::StoreOp>(loc, singleMapping.lookup(v), alloc);
    Value reloaded = parallelBuilder.create<fir::LoadOp>(loc, ty, alloc);
    rootMapping.map(v, reloaded);
    return alloc;
  };

  auto moveToSingle = [&](SingleRegion sr, OpBuilder allocaBuilder,
                          OpBuilder singleBuilder,
                          OpBuilder parallelBuilder) -> SmallVector<Value> {
    IRMapping singleMapping = rootMapping;
    SmallVector<Value> copyPrivate;

    for (Operation &op : llvm::make_range(sr.begin, sr.end)) {
      if (isSafeToParallelize(&op)) {
        singleBuilder.clone(op, singleMapping);
        parallelBuilder.clone(op, rootMapping);
      } else if (auto alloca = dyn_cast<fir::AllocaOp>(&op)) {
        auto hoisted =
            cast<fir::AllocaOp>(allocaBuilder.clone(*alloca, singleMapping));
        rootMapping.map(&*alloca, &*hoisted);
        rootMapping.map(alloca.getResult(), hoisted.getResult());
        copyPrivate.push_back(hoisted);
      } else {
        singleBuilder.clone(op, singleMapping);
        // Prepare reloaded values for results of operations that cannot be
        // safely parallelized and which are used after the region `sr`
        for (auto res : op.getResults()) {
          if (isTransitivelyUsedOutside(res, sr)) {
            auto alloc = mapReloadedValue(res, allocaBuilder, singleBuilder,
                                          parallelBuilder, singleMapping);
            if (alloc)
              copyPrivate.push_back(alloc);
          }
        }
      }
    }
    singleBuilder.create<omp::TerminatorOp>(loc);
    return copyPrivate;
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
        OpBuilder singleBuilder(sourceRegion.getContext());
        Block *singleBlock = new Block();
        singleBuilder.setInsertionPointToStart(singleBlock);

        OpBuilder allocaBuilder(sourceRegion.getContext());
        Block *allocaBlock = new Block();
        allocaBuilder.setInsertionPointToStart(allocaBlock);

        OpBuilder parallelBuilder(sourceRegion.getContext());
        Block *parallelBlock = new Block();
        parallelBuilder.setInsertionPointToStart(parallelBlock);

        omp::SingleOperands singleOperands;
        if (isLast)
          singleOperands.nowait = rootBuilder.getUnitAttr();
        singleOperands.copyprivateVars =
            moveToSingle(std::get<SingleRegion>(opOrSingle), allocaBuilder,
                         singleBuilder, parallelBuilder);
        cleanupBlock(singleBlock);
        for (auto var : singleOperands.copyprivateVars) {
          mlir::func::FuncOp funcOp =
              createCopyFunc(loc, var.getType(), firCopyFuncBuilder);
          singleOperands.copyprivateSyms.push_back(SymbolRefAttr::get(funcOp));
        }
        omp::SingleOp singleOp =
            rootBuilder.create<omp::SingleOp>(loc, singleOperands);
        singleOp.getRegion().push_back(singleBlock);
        rootBuilder.getInsertionBlock()->getOperations().splice(
            rootBuilder.getInsertionPoint(), parallelBlock->getOperations());
        targetRegion.front().getOperations().splice(
            singleOp->getIterator(), allocaBlock->getOperations());
        delete allocaBlock;
        delete parallelBlock;
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

  for (Block &targetBlock : targetRegion)
    cleanupBlock(&targetBlock);
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
  }
};
} // namespace
