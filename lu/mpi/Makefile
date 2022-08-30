CC = mpicc

all: main

main: main.c local_domain.c grid.c lu.c pivot.c
	$(CC) -o main main.c local_domain.c grid.c lu.c pivot.c

debug: main.c local_domain.c grid.c lu.c pivot.c
	$(CC) -o main main.c local_domain.c grid.c lu.c pivot.c -DDEBUG