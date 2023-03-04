EXE = elementwise_add
SRC = elementwise_add.cpp
OBJ = elementwise_add.o

CLROOT= ../Dependencies/

LDLIBS = -lm

CFLAG =  -Wall -fPIC
LDFLAG = -pie -fPIE
INC = -I$(CLROOT)
LIB = -L$(CLROOT)lib/ -lOpenCL

all: $(EXE)

$(EXE): $(OBJ)
	../Toolchain/bin/arm-linux-androideabi-g++ -o $@ $(LDFLAG) $^ $(LIB) -fPIE -lm

$(OBJ): $(SRC)
	../Toolchain/bin/arm-linux-androideabi-g++ -o $@ $(CFLAG) $(INC) -c $<

clean:
	rm -fr $(EXE) $(OBJ)

