// RUN: fir-opt --split-input-file --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

func.func @wsfunc() {
  %a = fir.alloca i32
  omp.parallel {
    omp.workshare {
      %t1 = "test.test1"() : () -> i32

      %c1 = arith.constant 1 : index
      %c42 = arith.constant 42 : index

      %c2 = arith.constant 2 : index
      "test.test3"(%c2) : (index) -> ()

      "omp.workshare_loop_wrapper"() ({
        omp.loop_nest (%arg1) : index = (%c1) to (%c42) inclusive step (%c1) {
          "test.test2"() : () -> ()
          omp.yield
        }
        omp.terminator
      }) : () -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL:   func.func @wsfunc() {
// CHECK:           %[[VAL_0:.*]] = fir.alloca i32
// CHECK:           omp.parallel {
// CHECK:             %[[VAL_1:.*]] = arith.constant true
// CHECK:             fir.if %[[VAL_1]] {
// CHECK:               omp.single {
// CHECK:                 %[[VAL_2:.*]] = "test.test1"() : () -> i32
// CHECK:                 %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:                 "test.test3"(%[[VAL_3]]) : (index) -> ()
// CHECK:                 omp.terminator
// CHECK:               }
// CHECK:               %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_5:.*]] = arith.constant 42 : index
// CHECK:               omp.wsloop nowait {
// CHECK:                 omp.loop_nest (%[[VAL_6:.*]]) : index = (%[[VAL_4]]) to (%[[VAL_5]]) inclusive step (%[[VAL_4]]) {
// CHECK:                   "test.test2"() : () -> ()
// CHECK:                   omp.yield
// CHECK:                 }
// CHECK:                 omp.terminator
// CHECK:               }
// CHECK:               omp.barrier
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

