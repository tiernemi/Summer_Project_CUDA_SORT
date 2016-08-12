CXX= g++
NVCC= /usr/local/cuda/bin/nvcc
LINK= nvcc

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall -lcuda $(PARALLEL) $(DEFINES)
CXXFLAGS    = -lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops -W -Wall -lcuda $(PARALLEL) $(DEFINES) -std=c++11 -lm

NVCCFLAGS	= -O5 -std=c++11 -arch=compute_35 -code=sm_35  --relocatable-device-code true --use_fast_math -Xptxas="-v" 

#--ptxas-options=-v -lineinfo #-maxrregcount 32
LIBS		= $(PARALLEL)
INCPATH		= /usr/include/

BIN = ./bin/
CXXSRCDIR = ./src/cpp_src/
CUSRCDIR = ./src/cu_src/
CPPINCDIR = ./inc/cpp_inc/
CUINCDIR = ./inc/cu_inc/
CUBPATH = ~/cub-1.5.2/cub/

####### Files
CXXFILES= $(shell ls $(CXXSRCDIR)*.cpp | xargs -n1 basename)
CUFILES= $(shell ls $(CUSRCDIR)*.cu | xargs -n1 basename)
CXXOBJS= $(CXXFILES:cpp=o)
CUOBJS= $(CUFILES:cu=o)

CPPINC= $(shell ls $(CPPINCDIR)*.hpp | xargs -n1 basename)
CUINC= $(shell ls $(CUINCDIR)*.hpp | xargs -n1 basename)

SOURCES=$(SRC)

CXXOBJECTS=$(addprefix $(BIN), $(CXXOBJS))
CUOBJECTS=$(addprefix $(BIN), $(CUOBJS))
OBJECTS= $(CXXOBJECTS) 
OBJECTS+= $(CUOBJECTS)

CPPHEADERS=$(addprefix $(CPPINCDIR), $(CPPINC))
CUHEADERS=$(addprefix $(CUINCDIR), $(CUINC))

CXXSRC= $(addprefix $(CXXSRCDIR), $CXXFILES)
CUSRC= $(addprefix $(CUSRCDIR), $CUFILES)
TARGET= sort

all: $(BIN) $(OBJECTS) 
	$(NVCC) $(OBJECTS) -o $(TARGET) -I$(INCPATH) -lcudadevrt
	chmod 777 $(OBJECTS)

.SECONDEXPANSION:
$(CXXOBJECTS): %.o: $$(addprefix $(CXXSRCDIR), $$(notdir %)).cpp $(HEADERS)
	$(CXX) -c $< $(CXXFLAGS) -I$(INCPATH) -I$(CPPINCDIR) -o $@ 

#easeperate compilation
.SECONDEXPANSION:
$(CUOBJECTS): %.o: $$(addprefix $(CUSRCDIR), $$(notdir %)).cu $(HEADERS)
	$(NVCC) -c $< $(NVCCFLAGS) -I$(CUBPATH)  -I$(INCPATH) -I$(CUINCDIR) -I$(CPPINCDIR) -o $@ 

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

