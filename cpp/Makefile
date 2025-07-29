# Top of Makefile
BUILD_DIR := build
TARGET_DIR := $(BUILD_DIR)/bin

# Test configuration
TEST_SRC_DIR := tests
TEST_SRC := $(shell find $(TEST_SRC_DIR) -name '*.cpp')
TEST_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(TEST_SRC))
GTEST_LIBS := -lgtest -lgtest_main -lpthread

# OpenCV detection
OPENCV_FLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# Safety-critical flags
SAFETY_FLAGS := -fstack-protector-strong
OPENCV_SAFE_FLAGS := -fexceptions -frtti

# Warning control
WARNING_FLAGS := -Wall -Wextra -Wpedantic -Werror
SUPPRESS_WARNINGS := -Wno-sign-compare -Wno-unused-variable -Wno-reorder

# Compiler setup
CXX := g++
CXXFLAGS := -std=c++17 $(WARNING_FLAGS) -O2 $(SUPPRESS_WARNINGS) $(OPENCV_FLAGS)
LDFLAGS := -pthread $(OPENCV_LIBS)

# Define safety-critical modules
SAFETY_CRITICAL_DIRS := core ml $(TEST_SRC_DIR)

# Per-directory flags
$(foreach dir,$(SAFETY_CRITICAL_DIRS),$(eval $(BUILD_DIR)/$(dir)/%.o: EXTRA_FLAGS := $(SAFETY_FLAGS) -fno-exceptions -fno-rtti))
$(foreach dir,vision apps,$(eval $(BUILD_DIR)/$(dir)/%.o: EXTRA_FLAGS := $(OPENCV_SAFE_FLAGS)))

# Source directories
SRC_DIRS := apps core ml $(TEST_SRC_DIR) vision
SRCS := $(shell find $(SRC_DIRS) -name '*.cpp')
OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# Main targets
TARGET := $(TARGET_DIR)/app
EXECUTABLES := $(TARGET_DIR)/main $(TARGET_DIR)/basicai

# BasicAI application
$(TARGET_DIR)/basicai: $(BUILD_DIR)/apps/basicai.o $(filter-out $(TEST_OBJS), $(OBJS))
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)


all: $(EXECUTABLES)

# Compilation rule
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -MMD -MP -c $< -o $@

# Main application
$(TARGET_DIR)/main: $(filter-out $(TEST_OBJS), $(OBJS))
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Test executable
$(TARGET_DIR)/tests: $(TEST_OBJS) $(filter-out $(BUILD_DIR)/apps/main.o, $(OBJS))
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(GTEST_LIBS)

run_tests: $(TARGET_DIR)/tests
	./$(TARGET_DIR)/tests

clean:
	rm -rf $(BUILD_DIR) $(TARGET_DIR)

.PHONY: all clean run_tests