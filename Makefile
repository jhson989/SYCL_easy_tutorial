CC = clang++ 
INCLUDE = -I/data/share/oneapi/llvm/build/include/sycl
CC_OPTION = -fsycl -fsycl-targets=nvptx64-nvidia-cuda
PROG = app 
FILES = vectorAdd.cpp

${PROG}: ${FILES} Makefile
	${CC} ${INCLUDE} ${CC_OPTION} -o app.out ${FILES}

clean:
	rm app.out

run:
	./app.out


    

