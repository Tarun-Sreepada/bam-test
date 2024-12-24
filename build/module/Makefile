
ifneq ($(KERNELRELEASE),)
	src := /export/home1/ltarun/bam-test/external/bam/module
	obj-m := bam.o
	bam-objs := pci.o list.o ctrl.o map.o
	ccflags-y += -D_CUDA -I/usr/src/nvidia-560.35.03/nvidia -I/export/home1/ltarun/bam-test/external/bam/src -I/export/home1/ltarun/bam-test/external/bam/src/linux
	KBUILD_EXTRA_SYMBOLS := /usr/src/nvidia-560.35.03/Module.symvers
else

.PHONY: default reload unload load clean install

default:
	$(MAKE) -C /lib/modules/5.15.0-122-generic/build M=/export/home1/ltarun/bam-test/build/module modules

clean:
	$(MAKE) -C /lib/modules/5.15.0-122-generic/build M=/export/home1/ltarun/bam-test/build/module clean

reload: unload load

unload:
	-rmmod bam.ko

load:
	insmod bam.ko max_num_ctrls=64

install: default
	$(MAKE) -C /lib/modules/5.15.0-122-generic/build M=/export/home1/ltarun/bam-test/build/module modules_install
	#$(MAKE) INSTALL_MOD_DIR=bam -C /lib/modules/5.15.0-122-generic/build M=/export/home1/ltarun/bam-test/build/module modules_install

endif

