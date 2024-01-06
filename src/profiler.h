#pragma once

#include <chrono>
#include <sstream>
#include <limits>
#include <unordered_map>

class Profiler {
 public:
  using Value = double;
  struct Record {
    int count = 0;
    // Time in seconds.
    Value total = 0;
    Value min = std::numeric_limits<Value>::max();
    Value max = -std::numeric_limits<Value>::max();

    friend std::string to_string(const Record& rec) {
      std::stringstream out;
      auto p = [&](Value val) {
        for (auto s : {" s", " ms", " us"}) {
          if (std::abs(val) >= 10) {
            out << val;
            return s;
          }
          val *= 1e3;
        }
        out << val;
        return " ns";
      };
      out.precision(0);
      out << std::fixed;
      out << "count=" << rec.count;
      out << ",  total=" << p(rec.total);
      out << ",  avg=" << p(rec.total / rec.count);
      out << ",  min=" << p(rec.min);
      out << ",  max=" << p(rec.max);
      return out.str();
    }
  };
  class Timer {
   public:
    Timer(Profiler* profiler, const std::string& name)
        : profiler_(profiler),
          name_(name),
          start_(std::chrono::steady_clock::now()) {}
    Timer(const Timer&) = delete;
    ~Timer() {
      auto delta = std::chrono::steady_clock::now() - start_;
      auto nsdur = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
      auto ns = nsdur.count();
      profiler_->AddRecord(name_, ns / 1e9);
    }
    Profiler* profiler_;

   private:
    std::string name_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
  };
  static Profiler* GetInstance() {
    static Profiler profiler;
    return &profiler;
  }
  void AddRecord(const std::string& name, Value time) {
    auto& rec = records_[name];
    ++rec.count;
    rec.total += time;
    if (time < rec.min) {
      rec.min = time;
    }
    if (time > rec.max) {
      rec.max = time;
    }
  }
  Timer MakeTimer(const std::string& name) {
    return Timer(this, name);
  }
  const auto& GetRecords() const {
    return records_;
  }
  void Clear() {
    records_.clear();
  }

 private:
  std::unordered_map<std::string, Record> records_;
};
