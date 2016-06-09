# UNIT tests nakefile

NVCC= /usr/local/cuda/bin/nvcc

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CXXFLAGS    = -lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops -W -Wall -lcuda $(PARALLEL) $(DEFINES) -std=c++11 -lm
NVCCFLAGS	= -O5 -DWITH_MY_DEBUG -std=c++11 -arch=sm_35 --relocatable-device-code true -lcudadevrt --use_fast_math 
LIBS		= $(PARALLEL)
INCPATH		= /usr/include/

BIN = ../bin/
CXXSRCDIR = ../src/cpp_src/
CUSRCDIR = ../src/cu_src/
INCDIR = ../inc/

####### Files
CXXFILES= $(shell ls $(CXXSRCDIR)*.cpp | xargs -n1 basename)
CXXFILES := $(filter-out main.cpp, $(CXXFILES))
CUFILES= $(shell ls $(CUSRCDIR)*.cu | xargs -n1 basename)
CXXOBJS= $(CXXFILES:cpp=o)
CUOBJS= $(CUFILES:cu=o)

CXXOBJECTS=$(addprefix $(BIN), $(CXXOBJS))
CUOBJECTS=$(addprefix $(BIN), $(CUOBJS))
OBJECTS= $(CXXOBJECTS) 
OBJECTS+= $(CUOBJECTS)

all: t/wvtest

$(BIN):
	mkdir $(BIN)

wvtestmain.o: wvtestmain.cc
	g++ -c $< -D WVTEST_CONFIGURED -o $@ $(CXXFLAGS) -I.

wvtest.o: wvtest.cc
	g++ -c $< -D WVTEST_CONFIGURED -o $@ $(CXXFLAGS) -I.

wvtest.t.o: t/wvtest.t.cc
	g++ -c $< -D WVTEST_CONFIGURED -o $@ $(CXXFLAGS) -I.

t/wvtest: $(BIN) $(OBJECTS) wvtest.t.o wvtest.o wvtestmain.o
	$(NVCC) $(OBJECTS) wvtest.t.o wvtest.o wvtestmain.o -o $@ -I$(INCDIR) -lcudadevrt

runtests: all
	t/wvtest

test: all
	./wvtestrun $(MAKE) runtests

clean::
	rm -f *~ t/*~ *.o t/*.o t/wvtest
