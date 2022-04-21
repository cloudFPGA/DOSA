#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Dec 2019
#  *     Authors: FAB, WEI, NGL
#  *
#  *     Description:
#  *       Makefile to build the Sofware (SW) part of the ZRLMPI cFa
#  * 
#  *     Synopsis:
#  *       make [target]  (use 'make help' to list the targets)
#  * 

### --- 
SOURCES = dosa_infer.cpp
HEADS = dosa_infer.hpp
### ------------------------------------------
CC       = gcc
CPPFLAGS = -D_XOPEN_SOURCE=700
CFLAGS   = -pedantic -Wall  -pthread
CFLAGS  += --std=c++11
#CFLAGS += -std=c99 -Wextra -Wconversion -Werror
CFLAGS  += -g -fPIC
LIBS     = -lstdc++
#ZRLMPI_PATH =$(zrlmpiDir)/LIB/SW/
ZRLMPI_PATH =./LIB/
LIBS    += -I$(ZRLMPI_PATH)
LIBS    += -L$(ZRLMPI_PATH)
PROG     =dosa_infer_pass
SHLIB    =dosa_infer_pass.so
BUILD_TYPE = debug # oder production
SOURCES += $(ZRLMPI_PATH)/ZRLMPI.cpp $(ZRLMPI_PATH)/zrlmpi_common.cpp
OBJS    = $(SOURCES:%.cpp=%.o)
HAEADS  += $(SOURCES:%.cpp=%.hpp)

#CFLAGS +=-DDEBUG -DZRLMPI_SW_ONLY
# ------------------------------------------
#$(PROG): MPI.hpp zrlmpi_common.hpp zrlmpi_int.hpp test.hpp
#MPI.o: MPI.cpp test.hpp zrlmpi_common.hpp MPI.hpp

.PHONY: all clean help lib


all: $(PROG) ## Builds the Software part of ZRLMPI to communicate with the FPGAs

lib: $(SHLIB) ## Builds the Software part of ZRLMPI as shared library

clean:  ## cleans all builds 
	rm -rf $(PROG) *.o $(SHLIB)

$(PROG): $(OBJS) $(HEADS)
	$(CC) -o $@ $(LDFLAGS) $(OBJS) $(LIBS)
ifeq ($(BUILD_TYPE), production)
	strip $@
endif 

$(SHLIB): $(OBJS) $(HEADS)
	$(CC) -o $@ $(LDFLAGS) $(OBJS) $(LIBS) -shared
ifeq ($(BUILD_TYPE), production)
	strip $@
endif 

lib_test.so: dosa_infer.cpp $(HEADS)
	@# rebuild dosa_infer.o with right flag
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $(LIBS) $< -o dosa_infer.o -DWRAPPER_TEST
	$(CC) -o $@ $(LDFLAGS) $(OBJS) $(LIBS) -shared -DWRAPPER_TEST


%.o: %.cpp $(foreach a,%.hpp,$(if $(findstring irrelevant,$a),,$a))
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $(LIBS) $< -o $@


print-%: ## A little make receipt to print a variable (usage example --> 'make print-SHELL_DIR')
	@echo $* = $($*)

help: ## Shows this help message
    # This target is for self documentation of the Makefile. 
    # Every text starting with '##' and placed after a target will be considered as helper text.
	@echo
	@echo 'Usage:'
	@echo '    make [target]'
	@echo	
	@echo 'Targets:'
	@cat ${MAKEFILE_LIST}  | egrep '^(.+)\:(.+)##\ (.+)' |  column -s ':#' |  sed 's/:.*#/||/' |  column -s '||' -t | sed -e 's/^/    /' 
	@echo
