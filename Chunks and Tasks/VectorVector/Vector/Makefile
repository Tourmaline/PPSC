TOPDIR=../..
include $(TOPDIR)/Makefile.include

INCLUDES=-I.

MAKEDEPEND= $(CC) -M $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $(CHTTLINCL) $(CHTINCL) $(CHTMLINCL) $(LEAF_MAT_INCL) -o $*.d $<

SRCS = 

OBJS= $(SRCS:.cc=.o)

all: libs

libs: libchtvec.a

libchtvec.a: $(OBJS)
	ar rcs libchtvec.a $(OBJS)

%.o: %.cc
	@$(MAKEDEPEND)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $(CHTTLINCL)  $(CHTINCL) $(CHTMLINCL) $(LEAF_MAT_INCL) -c $< -o $@

-include $(SRCS:.cc=.d)

check:
	
clean:
	rm -f *.a *.o *.d
