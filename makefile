CXX= g++
NVCC= /usr/local/cuda/bin/nvcc
LINK= nvcc

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall -lcuda $(PARALLEL) $(DEFINES)
CXXFLAGS    = -lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops -W -Wall -lcuda $(PARALLEL) $(DEFINES) -std=c++11 -lm

NVCCFLAGS	= -O5 -DWITH_MY_DEBUG -std=c++11 -arch=compute_35 -code=sm_35  --relocatable-device-code true -lcudadevrt --use_fast_math 
LIBS		= $(PARALLEL)
INCPATH		= /usr/include/

BIN = ./bin/
CXXSRCDIR = ./src/cpp_src/
CUSRCDIR = ./src/cu_src/
INCDIR = ./inc/

####### Files
CXXFILES= $(shell ls $(CXXSRCDIR)*.cpp | xargs -n1 basename)
CUFILES= $(shell ls $(CUSRCDIR)*.cu | xargs -n1 basename)
CXXOBJS= $(CXXFILES:cpp=o)
CUOBJS= $(CUFILES:cu=o)

INC= $(shell ls $(INCDIR)*.hpp | xargs -n1 basename)

SOURCES=$(SRC)
CXXOBJECTS=$(addprefix $(BIN), $(CXXOBJS))
CUOBJECTS=$(addprefix $(BIN), $(CUOBJS))
OBJECTS= $(CXXOBJECTS) 
OBJECTS+= $(CUOBJECTS)
HEADERS=$(addprefix $(INCDIR), $(INC))
CXXSRC= $(addprefix $(CXXSRCDIR), $CXXFILES)
CUSRC= $(addprefix $(CUSRCDIR), $CUFILES)
TARGET= sort

all: $(BIN) $(OBJECTS) 
	$(NVCC) $(OBJECTS) -o $(TARGET) -I$(INCPATH) -lcudadevrt
	chmod 777 $(OBJECTS)

.SECONDEXPANSION:
$(CXXOBJECTS): %.o: $$(addprefix $(CXXSRCDIR), $$(notdir %)).cpp $(HEADERS)
	$(CXX) -c $< $(CXXFLAGS) -I$(INCPATH) -o $@

#easeperate compilation
.SECONDEXPANSION:
$(CUOBJECTS): %.o: $$(addprefix $(CUSRCDIR), $$(notdir %)).cu $(HEADERS)
	$(NVCC) -c $< $(NVCCFLAGS) -I$(INCPATH) -o $@ 

$(BIN):
	mkdir $(BIN)

clean:
	rm -rf $(BIN) $(TARGET)

test: all
	./sort -f data/testDataP2.txt

unit_tests: all
	make -C unit_tests test

benchs: all
	make -C benchmarks bench

