#pragma once

#include <memory>
#include <string>
#include <unordered_map>

class Profiler {
 public:
  using Value = double;
  struct Record {
    Record();

    // Number fo calls.
    int count;
    // Time in seconds.
    Value total;
    Value min;
    Value max;
  };
  class Timer {
   public:
    Timer(Profiler* profiler, const std::string& name);
    Timer(const Timer&) = delete;
    ~Timer();

   private:
    struct Imp;
    const std::unique_ptr<Imp> imp;
  };
  static Profiler* GetInstance() {
    static Profiler profiler;
    return &profiler;
  }
  void AddRecord(const std::string& name, Value time);
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

std::string to_string(const Profiler::Record& rec);
