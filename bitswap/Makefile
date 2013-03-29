CXXC=i686-linux-g++

CXXFLAGS=-c -O4 -march=i686 -Wall

INCPATH= \
        -I. \

LIBS= \

LDFLAGS= \

CXXSOURCES= \
        main.cpp

CXXOBJECTS=$(CXXSOURCES:.cpp=.o)

EXECUTABLE=../../bin/test_bitswap

all: $(CXXSOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(CXXOBJECTS)
	$(CXXC) $(LIBS) $(LDFLAGS) $(CXXOBJECTS) -o $@

.cpp.o:
	$(CXXC) $(CXXFLAGS) $(INCPATH) $< -o $@

clean:
	-rm -f $(CXXOBJECTS) $(EXECUTABLE)
