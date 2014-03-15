SUBDIRS= Vector \
	time_vector\
	test_MatVecMul_vec\
test_MatVecMul_mat	
all:
	list='$(SUBDIRS)'; \
	for subdir in $$list; do \
	(cd $$subdir && $(MAKE)) \
	|| exit 1; \
	done; \
	echo; \
	echo 'Updated all targets in directories: $(SUBDIRS)'; \
	echo;
check:
	list='$(SUBDIRS)'; \
	for subdir in $$list; do \
	(cd $$subdir && $(MAKE) check) \
	|| exit 1; \
	done; \
	echo; \
	echo 'Successfully completed all tests in $(SUBDIRS)'; \
	echo;
clean:	
	(cd BasicVector && $(MAKE) clean);\
	list='$(SUBDIRS)'; \
	for subdir in $$list; do \
	(cd $$subdir && $(MAKE) clean) \
	done;
