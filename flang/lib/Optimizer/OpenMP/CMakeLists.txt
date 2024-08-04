get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FlangOpenMPTransforms
  LowerWorkshare.cpp

  DEPENDS
  FIRDialect
  FlangOpenMPPassesIncGen
  ${dialect_libs}

  LINK_LIBS
  FIRAnalysis
  FIRDialect
  FIRBuilder
  FIRDialectSupport
  FIRSupport
  FIRTransforms
  HLFIRDialect
  MLIRIR
  ${dialect_libs}

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
