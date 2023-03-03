EXE = gemm
SRC = gemm.cpp
OBJ = gemm.o

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

