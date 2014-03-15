TOPDIR=../..
COMMON=../
include $(TOPDIR)/Makefile.include

MAKEDEPEND= $(CC) -M $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CHTMLINCL) $(CHTTLINCL) $(CHTINCL) $(LEAF_MAT_INCL) -o $*.d $<
SRCS = registration.cc test_MatMatMul.cc

all: test_MatMatMul
	

# WORKER (only needed if BUILD_AS_SINGLE_PROGRAM == 0)
cht_worker: $(COMMON)/registration.o $(CHTMLDIR)/libchtml.a $(CHTTLDIR)/libchttl.a $(CHTDIR)/libcht.a $(BSMDIR)/libbsm.a
ifeq ($(BUILD_AS_SINGLE_PROGRAM),0)
	$(CC)  $(CFLAGS) $(CHTMLINCL) $(CHTTLINCL) $(CHTINCL) $(LEAF_MAT_INCL) -o $@ $(COMMON)/registration.o -Wl,-whole-archive $(CHTMLDIR)/libchtml.a $(CHTTLDIR)/libchttl.a -Wl,-no-whole-archive $(CHTDIR)/libcht.a $(BLASFLAGS) $(LDFLAGS) -lrt
else

endif

test_MatMatMul: test_MatMatMul_manager cht_worker
	
test_MatMatMul_manager: test_MatMatMul.o $(COMMON)/registration.o $(CHTMLDIR)/libchtml.a $(CHTTLDIR)/libchttl.a $(CHTDIR)/libcht.a $(BSMDIR)/libbsm.a
	$(CC) $(CFLAGS) $(CHTMLINCL) $(CHTTLINCL) $(CHTINCL) $(LEAF_MAT_INCL) -o $@ test_MatMatMul.o $(COMMON)/registration.o -Wl,-whole-archive $(CHTMLDIR)/libchtml.a $(CHTTLDIR)/libchttl.a -Wl,-no-whole-archive $(CHTDIR)/libcht.a $(BLASFLAGS) $(LDFLAGS) -lrt

%.o: %.cc
	@$(MAKEDEPEND)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CHTMLINCL) $(CHTTLINCL) $(CHTINCL) $(LEAF_MAT_INCL) -c $< -o $@

-include $(SRCS:.cc=.d)

check: all
	$(EXEC_PREFIX) ./test_MatMatMul_manager
	rm -f output.txt

clean:
	rm -f test_MatMatMul_manager cht_worker *.o *.d

