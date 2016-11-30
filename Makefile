CC=g++

EIGEN_PATH=/Users/EliFriedman/Developer/dynet-base/eigen
DYNET_PATH=/Users/EliFriedman/Developer/dynet-base/dynet

FLAGS=-g --std=c++11
INCLUDES=-I. -I$(EIGEN_PATH) -I$(DYNET_PATH) -I$(DYNET_PATH)/build/dynet
LINKS=-ldynet

exec: test.cpp characters.o corpus.o model.o
	$(CC) $(FLAGS) $(INCLUDES) $(LINKS) $^ -o exec

# exec: model.cpp characters.o corpus.o 
# 	$(CC) $(FLAGS) $(INCLUDES) $(LINKS) $^ -o exec

model.o: model.cpp model.h
	$(CC) $(FLAGS) $(INCLUDES) $(LINKS) -c model.cpp

corpus.o: corpus.cpp corpus.h
	$(CC) $(FLAGS) $(INCLUDES) $(LINKS) -c corpus.cpp

characters.o: characters.cpp characters.h
	$(CC) $(FLAGS) $(INCLUDES) $(LINKS) -c characters.cpp

clean:
	rm -f exec *.o 

bilstm:
	$(CC) $(FLAGS) $(INCLUDES) $(LINKS) tag-bilstm.cpp -o bilstm
