# Name of the HLS core to build (inserted by DOSA)
ipName =haddoc_wrapper_test

# line 4: don't modify below, above lines are ignored by DOSA generation
appName =${ipName}

DEPS := $(shell find ./src/ -type f)
ifeq ($(usedRoleDir),to-be-defined)
	IP_GUARD=$(usedRole2Dir)/.ip_guard
	IP_GUARD_FORCE=$(usedRole2Dir)/.ip_guard_delete_to_force_tcl_export
else
	IP_GUARD=$(usedRoleDir)/.ip_guard
	IP_GUARD_FORCE=$(usedRoleDir)/.ip_guard_delete_to_force_tcl_export
endif

.PHONY: all csim cosim cosim_view assert_env syn_test report
all: assert_env ${ipName}_prj/solution1/impl/ip

# assert ENVIRONMENT
assert_env:
ifndef usedRoleDir
	@echo "ERROR: usedRoleDir MUST BE DEFINED as environment variable!" 
	@exit 1
endif

${ipName}_prj/solution1/impl/ip: $(DEPS)
	@date +%s > .tmp_stamp_3
	@rm -rf ${ipName}_prj
	export dosaIpName=${ipName}; export hlsSim=0; export hlsCoSim=0; export useWrapperTest=0; vivado_hls -f run_hls.tcl
	@cat ${ipName}_prj/solution1/syn/report/${appName}_csynth.rpt
	@echo "-" > .tmp_stamp_2
	@date +%s > .tmp_stamp_1
	@echo -n "HLS build time: "
	@/bin/bash -c "cat <(cat .tmp_stamp_* | tr '\n' ' ') <(echo '') | /usr/bin/bc -l || echo 'see .tmp_stamp files'"
	@touch $@
	touch $(IP_GUARD)
	@# to be sure...
	rm -f $(IP_GUARD_FORCE)

csim:
	@/bin/echo -e "---------------\n\tThis simulation works only with *one* iteration with [1,3,10,10] as input and [1,4,5,5] as output!\n\tSo the flag WRAPPER_TEST is set.\n\t(to use the actual parameters, use 'make csim_orig')\n---------------\n"
	export dosaIpName=${ipName}; export hlsSim=1; export hlsCoSim=0; export useWrapperTest=1; vivado_hls -f run_hls.tcl

syn_test:
	@rm -rf ${ipName}_prj
	export dosaIpName=${ipName}; export hlsSim=0; export hlsCoSim=0; export useWrapperTest=1; vivado_hls -f run_hls.tcl
	@cat ${ipName}_prj/solution1/syn/report/${appName}_csynth.rpt

csim_orig:
	export hlsSim=1; export hlsCoSim=0; export useWrapperTest=0; vivado_hls -f run_hls.tcl

cosim:
	export hlsSim=0; export hlsCoSim=1; vivado_hls -f run_hls.tcl

cosim_view:
	@/bin/echo -e "current_fileset\nopen_wave_database $(appName).wdb\n" > ./$(ipName)_prj/solution1/sim/verilog/open_wave.tcl
	cd ./$(ipName)_prj/solution1/sim/verilog/; vivado -source open_wave.tcl

report:
	@cat vivado_hls.log
	@cat ${ipName}_prj/solution1/syn/report/${appName}_csynth.rpt

clean:
	rm -rf ${ipName}_prj
	rm -f vivado*.log hs*.log


