# ParthSolverDev

Parth is a framework that provides fill-reducing orderings, which can be used with state-of-the-art Cholesky solvers such as MKL, Accelerate, and CHOLMOD. The goal of Parth is to improve the efficiency of sparse matrix factorizations by minimizing fill-in during the Cholesky decomposition process, making it suitable for high-performance scientific and engineering applications.

---

**Note:** This project is currently under construction. The codebase and documentation are my raw versions, originally created for the Parth paper. Over the coming month, I will actively improve the documentation and clean up the codebase to make it easier to use.

---

## Benchmark

### IPC Benchmark

To generate the benchmark used in the Parth paper, please refer to the [IPC benchmark generator repository](https://github.com/BehroozZare/parth-ipc-benchmark-generator.git) for matrix generation. It comes with detailed instructions and a Docker setup for easy matrix generation.

### Remeshing Benchmark

To generate the remeshing benchmark, first download the set of meshes from the [Oded Stein Meshes repository](https://github.com/odedstein/meshes). Also, please make sure to cite their repository if you use their meshes—they saved my life! You can then feed these meshes to the `PatchRemeshDemo.cpp` code. (I will provide detailed documentation and clean code soon. For now, I’m just drowning in tasks!)

**Notes for this version of the code:**

You can use the Docker setup provided in the IPC benchmark repository to build this code—the dependencies are the same.

CHOLMOD usage on macOS has some issues in this codebase because I modified the CHOLMOD source for some experiments, which led to some library linking problems. Please use Accelerate on macOS. On Linux, everything should work fine.

Honestly though, the code is almost well-structured! So you can still use it in its current state (hopefully 🙂).

---

## Citation

If you use Parth or build upon this work, please cite:

Behrooz Zarebavani, Danny M. Kaufman, David I.W. Levin, and Maryam Mehri Dehnavi. 2025. *Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns*. ACM Trans. Graph. 44, 4 (August 2025), 17 pages. [https://arxiv.org/pdf/2501.04011](https://arxiv.org/pdf/2501.04011)
