TOPDIR=../..
COMMON=../..
include $(TOPDIR)/Makefile.include
NAME=test_MatVecMul

MAKEDEPEND= $(CC) -M $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CHTMLINCL) $(CHTTLINCL) $(CHTINCL)  $(CHTVECINCL) $(LEAF_MAT_INCL) -o $*.d $<
SRCS = registration.cc $(NAME).cc

all: $(NAME)
	

# WORKER (only needed if BUILD_AS_SINGLE_PROGRAM == 0)
cht_worker: $(COMMON)/registration.o $(CHTMLDIR)/libchtml.a $(CHTTLDIR)/libchttl.a $(CHTDIR)/libcht.a $(BSMDIR)/libbsm.a $(CHTVECDIR)/libchtvec.a
ifeq ($(BUILD_AS_SINGLE_PROGRAM),0)
	$(CC)  $(CFLAGS) $(CHTMLINCL) $(CHTTLINCL) $(CHTINCL) $(CHTVECINCL) $(LEAF_MAT_INCL) -o $@ $(COMMON)/registration.o -Wl,-whole-archive $(CHTMLDIR)/libchtml.a $(CHTTLDIR)/libchttl.a -Wl,-no-whole-archive $(CHTDIR)/libcht.a $(BSMDIR)/libbsm.a $(BLASFLAGS) $(LDFLAGS)  $(CHTVECDIR)/libchtvec.a -lrt
else

endif

$(NAME): $(NAME)_manager cht_worker
	
$(NAME)_manager: $(NAME).o $(COMMON)/registration.o $(CHTMLDIR)/libchtml.a $(CHTTLDIR)/libchttl.a $(CHTDIR)/libcht.a $(BSMDIR)/libbsm.a  $(CHTVECDIR)/libchtvec.a
	$(CC) $(CFLAGS) $(CHTMLINCL) $(CHTTLINCL) $(CHTINCL) $(CHTVECINCL) $(LEAF_MAT_INCL) -o $@ $(NAME).o $(COMMON)/registration.o -Wl,-whole-archive $(CHTMLDIR)/libchtml.a $(CHTTLDIR)/libchttl.a -Wl,-no-whole-archive $(CHTDIR)/libcht.a $(BSMDIR)/libbsm.a $(BLASFLAGS) $(LDFLAGS)  $(CHTVECDIR)/libchtvec.a -lrt

%.o: %.cc
	@$(MAKEDEPEND)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CHTMLINCL) $(CHTTLINCL) $(CHTINCL)  $(CHTVECINCL) $(LEAF_MAT_INCL) -c $< -o $@

-include $(SRCS:.cc=.d)

check: all
	$(EXEC_PREFIX) ./$(NAME)_manager
	rm -f output.txt

clean:
	rm -f $(NAME)_manager cht_worker *.o *.d

