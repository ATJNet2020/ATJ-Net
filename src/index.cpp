#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <limits>


class Random {
public:
  Random(uint32_t seed=123456789) : x(seed) {}

  uint32_t fast_uint32() {
    // https://codingforspeed.com/using-faster-psudo-random-generator-xorshift/
    // https://experilous.com/1/blog/post/perfect-fast-random-floating-point-numbers
    uint32_t t;
    t = x ^ (x << 11);
    x = y; y = z; z = w;
    return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
  }

  uint32_t fast_uniform(uint32_t a=0, uint32_t b=1) {
    uint32_t r = fast_uint32();
    return a + r % (b - a);
  }

  float fast_float() {
    uint32_t r = fast_uint32();
    union {
      uint32_t i;
      float f;
    } pun = { 0x3F800000U | (r >> 9) };
    return pun.f - 1.0f;
  }

  private:
  uint32_t x;
  uint32_t y = 362436069;
  uint32_t z = 521288629;
  uint32_t w = 88675123;
};


extern "C" {

void build_key_index(
    const int* data, int n_data,
    int* start, int* end, int n_idx) {
  int p = 0, q = 0;
  for (int i = 0; i < n_idx; i++) {
    while (q < n_data && data[q] == i)
      q++;
    start[i] = p;
    end[i] = q;
    p = q;
  }
}

void fetch_key_index(
    const int* data,
    int* start, int* end, int n_idx,
    int* res, int n_res, int max_sample) {
  static Random rand;
  int k = 0;
  for (int i = 0; i < n_idx; i++) {
    int p = start[i], q = end[i];
    if (p > q) {
      printf("fetch_key_index p > q not right!\n");
      exit(1);
    }
    start[i] = k;
    if (max_sample < 0 || q - p <= max_sample) {
      for (int j = p; j < q; j++)
        res[k++] = data[j];
    } else {
      for (int j = 0; j < max_sample; j++)
        res[k++] = data[ rand.fast_uniform(p, q) ];
    }
    end[i] = k;
  }
  if (k != n_res) {
    printf("fetch_key_index n_res not right!\n");
    exit(1);
  }
}

void safe_index_int(
    const int* data, int n_data, int n_dim,
    const int* idx, int n_idx, int* res) {
  for (int i = 0; i < n_idx; i++) {
    if (idx[i] >= 0 && idx[i] < n_data)
      memcpy(res + i*n_dim, data + idx[i]*n_dim, sizeof(int)*n_dim);
    else
      memset(res + i*n_dim, 0, sizeof(int)*n_dim);
  }
}

void safe_index_float(
    const float* data, int n_data, int n_dim,
    const int* idx, int n_idx, float* res) {
  for (int i = 0; i < n_idx; i++) {
    if (idx[i] >= 0 && idx[i] < n_data)
      memcpy(res + i*n_dim, data + idx[i]*n_dim, sizeof(float)*n_dim);
    else
      memset(res + i*n_dim, 0, sizeof(float)*n_dim);
  }
}

void group_by(
    const int* start, const int* end, int n_idx, const int* idx,
    const float* target,
    float* mean, float* std, float* skew, float* kurt) {

  for (int i = 0; i < n_idx; i++)
    if (start[i] > end[i]) {
      printf("start %d > end %d when group_by.", start[i], end[i]);
      exit(1);

    } else if (start[i] == end[i]) {
      mean[i] = std[i] = skew[i] = kurt[i] = \
                std::numeric_limits<float>::quiet_NaN();

    } else {

      double sum1 = 0.0;
      int valid_n = 0;
      for (int j = start[i]; j < end[i]; j++) {
        double val = target[idx[j]];
        if (isfinite(val)) {
          sum1 += val;
          valid_n++;
        }
      }

      if (valid_n == 0) {
        mean[i] = std[i] = skew[i] = kurt[i] = \
                  std::numeric_limits<float>::quiet_NaN();
        continue;
      }

      mean[i] = sum1 / valid_n;

      double sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
      for (int j = start[i]; j < end[i]; j++) {
        double val = target[idx[j]];
        if (isfinite(val)) {
          double x = val - mean[i];
          sum2 += x*x;
          sum3 += x*x*x;
          sum4 += x*x*x*x;
        }
      }
      double var = sum2 / valid_n;
      std[i] = sqrt(var);
      if (valid_n == 1)
        skew[i] = 0.0;
      else
        skew[i] = sum3 / valid_n / var / std[i];
      if (valid_n <= 1)
        kurt[i] = -3.0;
      else
        kurt[i] = sum4 / valid_n / var / var - 3;
    }
}

void arange_keyindex(
    const int* start, const int* end, int* arr, int m, int n) {
  int cur = 0, k = 0;
  for (int i = 0; i < m; i++) {
    for (int j = start[i]; j < end[i]; j++)
      arr[cur++] = k++;
    int rest = n - (end[i] - start[i]);
    while (rest--)
      arr[cur++] = -1;
  }
  if (cur != m * n) {
    printf("arange_keyindex cur vs m vs n: %d %d %d\n", cur, m, n);
    exit(1);
  }
}

}
