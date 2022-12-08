############################################################
# makefile for citylbm
# support intel/gnu C++; OpenMP; OpenMPI
############################################################

include makefile.compiler.in
include makefile.macro.in
-include makefile.EigenG-Batched.in

# dir
SRC_DIR := src
SRCS := $(wildcard $(SRC_DIR)/*.cu)
vpath %.cu  $(SRC_DIR)

OBJ_DIR := obj
OBJS := $(subst $(SRC_DIR)/,$(OBJ_DIR)/,$(addsuffix .o,$(SRCS)))

RUN_DIR := run
TARGET := $(RUN_DIR)/a.out

.PHONY: all run clean resultclean resultcleanv tagfiles

all:
	$(MAKE) $(TARGET)

run: all
	mpirun -n 1 $(TARGET) | tee log/runlog.txt

bindiff: all
	make -C bindiff test_calc

$(TARGET): $(OBJS)
	$(LD) $^ -o $@ $(LDFLAGS)
	ldd $@

$(OBJ_DIR)/%.cu.o: %.cu
	@mkdir -p $(OBJ_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@

-include $(DEPENDS)

-include makefile.test.in

clean:
	@rm -fv $(TARGET)
	@rm -fv $(OBJS)

resultclean:
	@find io -name '*.dat' | xargs rm -f
	@find io -name '*.png' | xargs rm -f
	@find io -name '*.h5'  | xargs rm -f

resultcleanv:
	@find io -name '*.dat' | xargs rm -fv
	@find io -name '*.png' | xargs rm -fv
	@find io -name '*.h5'  | xargs rm -fv

tagfiles: # universal ctags
	ctags -R --langmap=c:+.hpp --langmap=c:+.cu $(SRC_DIR)

