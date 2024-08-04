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
// CHECK:             omp.single {
// CHECK:               %[[VAL_1:.*]] = "test.test1"() : () -> i32
// CHECK:               %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:               "test.test3"(%[[VAL_2]]) : (index) -> ()
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_4:.*]] = arith.constant 42 : index
// CHECK:             omp.wsloop nowait {
// CHECK:               omp.loop_nest (%[[VAL_5:.*]]) : index = (%[[VAL_3]]) to (%[[VAL_4]]) inclusive step (%[[VAL_3]]) {
// CHECK:                 "test.test2"() : () -> ()
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.barrier
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

