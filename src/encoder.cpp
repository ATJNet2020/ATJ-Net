#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include "Python.h"

bool cmp(const std::pair<std::string, int> a,
         const std::pair<std::string, int> b) {
  if (a.second == b.second)
    return a.first < b.first;
  else
    return a.second > b.second;
}

class Encoder {
private:
  std::vector<std::string> names;
  std::unordered_map<std::string, std::vector<std::string>> datas;
  std::unordered_map<std::string, int> counter;
  std::vector<std::pair<std::string, int>> vec;
  std::unordered_map<std::string, int> encoder;

public:
  void update(const char* name, PyObject* data[], int n) {
    names.push_back(std::string(name));
    datas[std::string(name)] = std::vector<std::string>(n);
    auto & cur_data = datas[std::string(name)];
    for (int i = 0; i < n; i++)
      if (PyUnicode_Check(data[i])) {
        // Py_ssize_t len;
        const char* p = PyUnicode_AsUTF8AndSize(data[i], NULL);
        if (p == NULL) {
          printf("Encoder update error at index:%d\n", i);
          exit(1);
        }
        std::string str(p);
        cur_data[i] = str;
        counter[str]++;
      }
  }

  void vectorize() {
    sort(names.begin(), names.end());
    vec.resize(counter.size());
    std::copy(counter.begin(), counter.end(), vec.begin());
    sort(vec.begin(), vec.end(), cmp);
  }

  int size() {
    return int(vec.size());
  }

  void save(const char* path) {
    FILE* file = fopen(path, "w");
    if (file == NULL) {
      printf("fopen file error: %s\n", path);
      exit(1);
    }

    fprintf(file, "{\"columns\": [");
    for (int i = 0; i < int(names.size()); i++)
      if (i == 0)
        fprintf(file, "\"%s\"", names[i].c_str());
      else
        fprintf(file, ", \"%s\"", names[i].c_str());
    fprintf(file, "], \"unique\": %d}\n\n", int(vec.size()));

    for (auto pair = vec.begin(); pair != vec.end(); pair++)
      fprintf(file, "%s\t%d\n", pair->first.c_str(), pair->second);

    fclose(file);
  }

  void encode() {
    int n = int(vec.size());
    for (int i = 0; i < n; i++)
      encoder[vec[i].first] = i+1;
  }

  void remap(const char* name, int* new_data, int n) {
    const auto & cur_data = datas[std::string(name)];
    for (int i = 0; i < n; i++) {
      if (cur_data[i].length() != 0)
        new_data[i] = encoder.at(cur_data[i]);
      else
        new_data[i] = 0;
    }
  }

};

extern "C" {

void* encoder_new() {
  return new Encoder();
}

void encoder_update(void* ptr, const char* name, PyObject* data[], int n) {
  Encoder* ref = reinterpret_cast<Encoder*>(ptr);
  ref->update(name, data, n);
}

void encoder_vectorize(void* ptr) {
  Encoder* ref = reinterpret_cast<Encoder*>(ptr);
  ref->vectorize();
}

int encoder_size(void* ptr) {
  Encoder* ref = reinterpret_cast<Encoder*>(ptr);
  return ref->size();
}

void encoder_save(void* ptr, const char* path) {
  Encoder* ref = reinterpret_cast<Encoder*>(ptr);
  ref->save(path);
}

void encoder_encode(void* ptr) {
  Encoder* ref = reinterpret_cast<Encoder*>(ptr);
  ref->encode();
}

void encoder_remap(void* ptr, const char* name, int* new_data, int n) {
  Encoder* ref = reinterpret_cast<Encoder*>(ptr);
  ref->remap(name, new_data, n);
}

void encoder_destroy(void* ptr) {
  Encoder* ref = reinterpret_cast<Encoder*>(ptr);
  delete ref;
}

}
