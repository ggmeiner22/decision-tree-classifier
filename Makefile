CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra -pedantic

INCLUDES = -Iinclude
SRCS = src/main.cpp src/Dataset.cpp src/DecisionTree.cpp
OBJS = $(SRCS:.cpp=.o)

all: dtree

dtree: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o dtree

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) dtree

.PHONY: all clean
