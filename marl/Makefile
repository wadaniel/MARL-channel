CHANNEL=run-vracer-multi 
KORALICXX=${CXX}
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.SECONDARY:
.PHONY: all
all: channel

channel: $(CHANNEL)

$(CHANNEL): %: %.o env.o 
	$(KORALICXX) -o $@ $^ $(KORALILIBS) 

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

env.o: _model/env.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) 

.PHONY: clean
clean:
	$(RM) $(CHANNEL) *.o 
