##################################################
# Sample Python Wrapping
##################################################

# Should include basic flags for Python boost bindings 
include ../../../../../MakefileBase

LIBRARY = ../_ifs_accu.so


##################################################
# Include your sources and flags here:
##################################################
SOURCES_C = 

SOURCES_CPP = \
py_ifs_accu.cpp

CFLAGS += -I${ESMERALDA}/python

OBJS_C=$(SOURCES_C:.c=.o)
OBJS_CPP=$(SOURCES_CPP:.cpp=.o)
OBJS =  $(OBJS_C) $(OBJS_CPP) 

.cpp.o:
	$(CXX) $(CXXFLAGS) $(CFLAGS) -c $< -o $@

.SUFFIXES:
.SUFFIXES:  .cpp .o

##################################################

.PHONY: all
all: $(LIBRARY)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(LIBRARY)

##################################################

$(LIBRARY): $(OBJS)
	$(CXX) $(LFLAGS) -o $@ $^ $(LLIBS)

##################################################
