CFLAGS = -Wall -Wextra -DLOCAL -Wshadow -Wno-unused-result -Wpointer-arith -Wcast-qual -Wunreachable-code
PYPATH = -L${HOME}/Lib/anaconda3/lib -I${HOME}/Lib/anaconda3/include/python3.7m -lpython3.7m

all:
	g++ -std=c++11 -shared index.cpp -o .index.so -fPIC $(CFLAGS) -O2
	g++ -std=c++11 -shared encoder.cpp -o .encoder.so -fPIC $(CFLAGS) $(PYPATH) -O2

preprocess:
	rm -rf ../data
	python preprocess.py ~/Dataset/table/a &
	python preprocess.py ~/Dataset/table/b &
	python preprocess.py ~/Dataset/table/c &
	python preprocess.py ~/Dataset/table/d &
	python preprocess.py ~/Dataset/table/e &
	python preprocess.py ~/Dataset/table/t &

freeze:
	rm -rf ../data/?/freeze/
	python engine.py --dataset_name a,b,c,d,e,t,x --is_brief

pack:
	tar -zcvf code.tar.gz automl/*.py *.py *.cpp .index.so makefile requirements.txt

clean:
	rm -f *.so
