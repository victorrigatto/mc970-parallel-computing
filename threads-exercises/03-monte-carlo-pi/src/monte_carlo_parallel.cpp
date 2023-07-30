#include <cassert>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

// Function to generate random numbers between -1 and 1
double random_number() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<double> dis(-1.0, 1.0);
  return dis(gen);
}

// Function to estimate pi using the Monte Carlo method
void calculate_pi(int &count, int num_iterations) {
  int hits = 0;
  for (int i = 0; i < num_iterations; ++i) {
    double x = random_number();
    double y = random_number();

    if (x * x + y * y <= 1.0) {
      ++hits;
    }
  }

  count += hits;

  std::cout << "hits: " << hits << " of " << num_iterations << std::endl;
}

int main() {
  const int num_iterations = 30000000;

  int count = 0;
  int num_threads = 4;
  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    int num_iterations_per_thread = num_iterations / num_threads;
    threads.emplace_back(calculate_pi, std::ref(count), num_iterations_per_thread);
  }

  // Aguarda todos os threads terminarem
  for (std::thread &thread : threads) {
    thread.join();
  }

  double pi = 4.0 * static_cast<double>(count) / static_cast<double>(num_iterations);
  std::cout << "Used " << num_iterations << " points to estimate pi: " << pi << std::endl;

  return 0;
}
