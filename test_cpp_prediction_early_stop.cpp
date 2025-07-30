#include <iostream>
#include <vector>
#include <cassert>

// Mock the LightGBM header structure
namespace LightGBM {
    struct PredictionEarlyStopConfig {
        int round_period;
        double margin_threshold;
    };

    struct PredictionEarlyStopInstance {
        std::function<bool(const double*, int)> callback_function;
        int round_period;
    };

    // Mock Log::Fatal for testing (instead of terminating, we'll throw)
    struct Log {
        static void Fatal(const char* msg) {
            throw std::runtime_error(msg);
        }
        template<typename... Args>
        static void Fatal(const char* format, Args... args) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), format, args...);
            throw std::runtime_error(buffer);
        }
    };
}

// Copy the implementation from prediction_early_stop.cpp
namespace LightGBM {

PredictionEarlyStopInstance CreateNone(const PredictionEarlyStopConfig&) {
  return PredictionEarlyStopInstance{
    [](const double*, int) {
    return false;
  },
    std::numeric_limits<int>::max()  // make sure the lambda is almost never called
  };
}

PredictionEarlyStopInstance CreateMulticlass(const PredictionEarlyStopConfig& config) {
  // margin_threshold will be captured by value
  const double margin_threshold = config.margin_threshold;

  return PredictionEarlyStopInstance{
    [margin_threshold](const double* pred, int sz) {
    if (sz < 2) {
      Log::Fatal("Multiclass early stopping needs predictions to be of length two or larger");
    }

    // copy and sort
    std::vector<double> votes(static_cast<size_t>(sz));
    for (int i = 0; i < sz; ++i) {
      votes[i] = pred[i];
    }
    std::partial_sort(votes.begin(), votes.begin() + 2, votes.end(), std::greater<double>());

    const auto margin = votes[0] - votes[1];

    if (margin > margin_threshold) {
      return true;
    }

    return false;
  },
    config.round_period
  };
}

PredictionEarlyStopInstance CreateBinary(const PredictionEarlyStopConfig& config) {
  // margin_threshold will be captured by value
  const double margin_threshold = config.margin_threshold;

  return PredictionEarlyStopInstance{
    [margin_threshold](const double* pred, int sz) {
    if (sz != 1) {
      Log::Fatal("Binary early stopping needs predictions to be of length one");
    }
    const auto margin = 2.0 * fabs(pred[0]);

    if (margin > margin_threshold) {
      return true;
    }

    return false;
  },
    config.round_period
  };
}

PredictionEarlyStopInstance CreatePredictionEarlyStopInstance(const std::string& type,
                                                              const PredictionEarlyStopConfig& config) {
  if (type == "none") {
    return CreateNone(config);
  } else if (type == "multiclass") {
    return CreateMulticlass(config);
  } else if (type == "binary") {
    return CreateBinary(config);
  } else {
    Log::Fatal("Unknown early stopping type: %s", type.c_str());
  }

  // Fix for compiler warnings about reaching end of control
  return CreateNone(config);
}

}  // namespace LightGBM

// Test cases
int main() {
    using namespace LightGBM;
    
    std::cout << "=== C++ Prediction Early Stop Tests ===" << std::endl;
    
    // Test 1: None type
    {
        PredictionEarlyStopConfig config{5, 0.5};
        auto instance = CreatePredictionEarlyStopInstance("none", config);
        
        std::vector<double> pred1 = {0.1, 0.9};
        std::vector<double> pred2 = {0.99, 0.01};
        
        bool result1 = instance.callback_function(pred1.data(), pred1.size());
        bool result2 = instance.callback_function(pred2.data(), pred2.size());
        
        std::cout << "None type - pred1: " << result1 << " (expected: 0)" << std::endl;
        std::cout << "None type - pred2: " << result2 << " (expected: 0)" << std::endl;
        std::cout << "None type - round_period: " << instance.round_period << " (expected: " << std::numeric_limits<int>::max() << ")" << std::endl;
        
        assert(result1 == false);
        assert(result2 == false);
        assert(instance.round_period == std::numeric_limits<int>::max());
    }
    
    // Test 2: Multiclass type
    {
        PredictionEarlyStopConfig config{5, 0.5};
        auto instance = CreatePredictionEarlyStopInstance("multiclass", config);
        
        // Test cases where margin > threshold (should stop)
        std::vector<double> pred1 = {0.8, 0.2};  // margin = 0.6 > 0.5
        std::vector<double> pred2 = {0.9, 0.1, 0.0};  // margin = 0.8 > 0.5
        
        // Test cases where margin <= threshold (should not stop)
        std::vector<double> pred3 = {0.6, 0.4};  // margin = 0.2 <= 0.5
        std::vector<double> pred4 = {0.75, 0.25};  // margin = 0.5 = 0.5 (not >)
        
        bool result1 = instance.callback_function(pred1.data(), pred1.size());
        bool result2 = instance.callback_function(pred2.data(), pred2.size());
        bool result3 = instance.callback_function(pred3.data(), pred3.size());
        bool result4 = instance.callback_function(pred4.data(), pred4.size());
        
        std::cout << "Multiclass - pred1 (0.8,0.2): " << result1 << " (expected: 1)" << std::endl;
        std::cout << "Multiclass - pred2 (0.9,0.1,0.0): " << result2 << " (expected: 1)" << std::endl;
        std::cout << "Multiclass - pred3 (0.6,0.4): " << result3 << " (expected: 0)" << std::endl;
        std::cout << "Multiclass - pred4 (0.75,0.25): " << result4 << " (expected: 0)" << std::endl;
        std::cout << "Multiclass - round_period: " << instance.round_period << " (expected: 5)" << std::endl;
        
        assert(result1 == true);
        assert(result2 == true);
        assert(result3 == false);
        assert(result4 == false);
        assert(instance.round_period == 5);
        
        // Test edge case - single prediction (should throw)
        std::vector<double> pred_single = {0.5};
        try {
            instance.callback_function(pred_single.data(), pred_single.size());
            assert(false);  // Should not reach here
        } catch (const std::runtime_error& e) {
            std::cout << "Multiclass - single pred correctly threw: " << e.what() << std::endl;
        }
    }
    
    // Test 3: Binary type
    {
        PredictionEarlyStopConfig config{3, 1.0};
        auto instance = CreatePredictionEarlyStopInstance("binary", config);
        
        // Test cases where 2*|pred| > threshold (should stop)
        std::vector<double> pred1 = {0.6};  // 2*0.6 = 1.2 > 1.0
        std::vector<double> pred2 = {-0.7}; // 2*0.7 = 1.4 > 1.0
        
        // Test cases where 2*|pred| <= threshold (should not stop)
        std::vector<double> pred3 = {0.4};  // 2*0.4 = 0.8 <= 1.0
        std::vector<double> pred4 = {0.5};  // 2*0.5 = 1.0 = 1.0 (not >)
        std::vector<double> pred5 = {-0.3}; // 2*0.3 = 0.6 <= 1.0
        
        bool result1 = instance.callback_function(pred1.data(), pred1.size());
        bool result2 = instance.callback_function(pred2.data(), pred2.size());
        bool result3 = instance.callback_function(pred3.data(), pred3.size());
        bool result4 = instance.callback_function(pred4.data(), pred4.size());
        bool result5 = instance.callback_function(pred5.data(), pred5.size());
        
        std::cout << "Binary - pred1 (0.6): " << result1 << " (expected: 1)" << std::endl;
        std::cout << "Binary - pred2 (-0.7): " << result2 << " (expected: 1)" << std::endl;
        std::cout << "Binary - pred3 (0.4): " << result3 << " (expected: 0)" << std::endl;
        std::cout << "Binary - pred4 (0.5): " << result4 << " (expected: 0)" << std::endl;
        std::cout << "Binary - pred5 (-0.3): " << result5 << " (expected: 0)" << std::endl;
        std::cout << "Binary - round_period: " << instance.round_period << " (expected: 3)" << std::endl;
        
        assert(result1 == true);
        assert(result2 == true);
        assert(result3 == false);
        assert(result4 == false);
        assert(result5 == false);
        assert(instance.round_period == 3);
        
        // Test edge case - multiple predictions (should throw)
        std::vector<double> pred_multi = {0.5, 0.3};
        try {
            instance.callback_function(pred_multi.data(), pred_multi.size());
            assert(false);  // Should not reach here
        } catch (const std::runtime_error& e) {
            std::cout << "Binary - multiple pred correctly threw: " << e.what() << std::endl;
        }
    }
    
    // Test 4: Unknown type (should throw)
    {
        PredictionEarlyStopConfig config{1, 0.0};
        try {
            auto instance = CreatePredictionEarlyStopInstance("unknown", config);
            assert(false);  // Should not reach here
        } catch (const std::runtime_error& e) {
            std::cout << "Unknown type correctly threw: " << e.what() << std::endl;
        }
    }
    
    std::cout << "All C++ tests passed!" << std::endl;
    return 0;
}