execute_process(COMMAND "${OPS_CLI}" gemm-check --m 1junk --n 1 --k 1
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(code EQUAL 0 OR NOT "${output}${error}" MATCHES "invalid numeric value")
  message(FATAL_ERROR "Malformed numeric option was not rejected by parser: ${output}${error}")
endif()
