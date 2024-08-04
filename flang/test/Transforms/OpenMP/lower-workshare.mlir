// RUN: fir-opt --lower-workshare %s | FileCheck %s

module {
// CHECK-LABEL:   func.func @simple(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_4]] x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.heap<!fir.array<42xi32>>
// CHECK:           omp.parallel {
// CHECK:             omp.single {
// CHECK:               %[[VAL_7:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:               %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_7]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:               %[[VAL_9:.*]] = builtin.unrealized_conversion_cast %[[VAL_8]]#0 : !fir.ref<!fir.array<42xi32>> to !llvm.ptr
// CHECK:               llvm.store %[[VAL_9]], %[[VAL_5]] : !llvm.ptr, !llvm.ptr
// CHECK:               %[[VAL_10:.*]] = fir.allocmem !fir.array<42xi32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:               %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_7]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<42xi32>>, !fir.heap<!fir.array<42xi32>>)
// CHECK:               fir.store %[[VAL_11]]#0 to %[[VAL_6]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             %[[VAL_12:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> !llvm.ptr
// CHECK:             %[[VAL_13:.*]] = builtin.unrealized_conversion_cast %[[VAL_12]] : !llvm.ptr to !fir.ref<!fir.array<42xi32>>
// CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<!fir.array<42xi32>>>
// CHECK:             omp.wsloop {
// CHECK:               omp.loop_nest (%[[VAL_15:.*]]) : index = (%[[VAL_1]]) to (%[[VAL_3]]) inclusive step (%[[VAL_1]]) {
// CHECK:                 %[[VAL_16:.*]] = hlfir.designate %[[VAL_13]] (%[[VAL_15]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:                 %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
// CHECK:                 %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_2]] : i32
// CHECK:                 %[[VAL_19:.*]] = hlfir.designate %[[VAL_14]] (%[[VAL_15]])  : (!fir.heap<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:                 hlfir.assign %[[VAL_18]] to %[[VAL_19]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.barrier
// CHECK:             omp.single nowait {
// CHECK:               hlfir.assign %[[VAL_14]] to %[[VAL_13]] : !fir.heap<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
// CHECK:               fir.freemem %[[VAL_14]] : !fir.heap<!fir.array<42xi32>>
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.barrier
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
  func.func @simple(%arg0: !fir.ref<!fir.array<42xi32>>) {
    omp.parallel {
      omp.workshare {
        %c42 = arith.constant 42 : index
        %c1_i32 = arith.constant 1 : i32
        %0 = fir.shape %c42 : (index) -> !fir.shape<1>
        %1:2 = hlfir.declare %arg0(%0) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
        %2 = fir.allocmem !fir.array<42xi32> {bindc_name = ".tmp.array", uniq_name = ""}
        %3:2 = hlfir.declare %2(%0) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<42xi32>>, !fir.heap<!fir.array<42xi32>>)
        %true = arith.constant true
        %c1 = arith.constant 1 : index
        omp.wsloop {
          omp.loop_nest (%arg1) : index = (%c1) to (%c42) inclusive step (%c1) {
            %7 = hlfir.designate %1#0 (%arg1)  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
            %8 = fir.load %7 : !fir.ref<i32>
            %9 = arith.subi %8, %c1_i32 : i32
            %10 = hlfir.designate %3#0 (%arg1)  : (!fir.heap<!fir.array<42xi32>>, index) -> !fir.ref<i32>
            hlfir.assign %9 to %10 temporary_lhs : i32, !fir.ref<i32>
            omp.yield
          }
          omp.terminator
        }
        %4 = fir.undefined tuple<!fir.heap<!fir.array<42xi32>>, i1>
        %5 = fir.insert_value %4, %true, [1 : index] : (tuple<!fir.heap<!fir.array<42xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<42xi32>>, i1>
        %6 = fir.insert_value %5, %3#0, [0 : index] : (tuple<!fir.heap<!fir.array<42xi32>>, i1>, !fir.heap<!fir.array<42xi32>>) -> tuple<!fir.heap<!fir.array<42xi32>>, i1>
        hlfir.assign %3#0 to %1#0 : !fir.heap<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
        fir.freemem %3#0 : !fir.heap<!fir.array<42xi32>>
        omp.terminator
      }
      omp.terminator
    }
    return
  }
}
