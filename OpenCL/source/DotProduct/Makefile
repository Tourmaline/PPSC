INCL=-I/opt/AMDAPP/include
LIB=-L/opt/AMDAPP/lib/x86_64

DEPS=../../common/*h 

NAME=DotProduct

all: $(NAME)

$(NAME): $(NAME).o 
	g++ $(NAME).o -o $(NAME) $(INCL) $(LIB) -O3 -lOpenCL -lrt -Wall

$(NAME).o: $(NAME).cpp $(DEPS)
	g++ -c $(NAME).cpp  $(INCL) $(LIB) -O3 -lOpenCL -lrt -Wall

clean:
	rm -rf $(NAME).o $(NAME)
	rm -rf ../kernels/$(NAME)_kernel.cl.bin
