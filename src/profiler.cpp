#include "profiler.h"

#include <chrono>
#include <limits>
#include <sstream>

Profiler::Record::Record()
    : count(0),
      total(0),
      min(std::numeric_limits<Value>::max()),
      max(-std::numeric_limits<Value>::max()) {}

struct Profiler::Timer::Imp {
  Imp(Profiler* profiler, const std::string& name)
      : profiler_(profiler),
        name_(name),
        start_(std::chrono::steady_clock::now()) {}

  ~Imp() {
    auto delta = std::chrono::steady_clock::now() - start_;
    auto nsdur = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
    auto ns = nsdur.count();
    profiler_->AddRecord(name_, ns / 1e9);
  }

  Profiler* profiler_;
  std::string name_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
};

void Profiler::AddRecord(const std::string& name, Value time) {
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

Profiler::Timer::Timer(Profiler* profiler, const std::string& name)
    : imp(new Imp(profiler, name)) {}

Profiler::Timer::~Timer() = default;

std::string to_string(const Profiler::Record& rec) {
  std::stringstream out;
  auto p = [&](Profiler::Value val) {
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
