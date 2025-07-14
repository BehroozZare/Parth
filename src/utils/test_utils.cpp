//
// Created by Kazem on 10/11/19.
//
#include <random>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include "includes/test_utils.h"
namespace sym_lib
{

	double double_rand(double d_min, double d_max)
	{
		double f = (double)rand() / RAND_MAX;
		return d_min + f * (d_max - d_min);
	}

	void generate_uniq_rand_vector(int m, int n, std::vector<int> &rand_vec,
								   unsigned seed)
	{
		std::vector<int> numbers;
		for (int i = 0; i < n; i++)
			numbers.push_back(i);
		if (seed == 0)
			seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(numbers.begin(), numbers.end(), std::default_random_engine(seed));
		for (int j = 0; j < m; ++j)
		{
			rand_vec.push_back(numbers[j]);
		}
	}

	CSC *random_square_sparse(size_t n, double density, double max_val,
							  unsigned seed)
	{
		// Warning: for large n, e.g., n>5e5, this is very slow
		int nnz_per_col = density * n + 1;
		nnz_per_col = nnz_per_col > n ? n : nnz_per_col;
		size_t nnz = n * nnz_per_col;
		if (nnz > INT32_MAX)
			return NULLPNTR;
		CSC *A = new CSC(n, n, nnz);
		A->p[0] = 0;
		for (int i = 0; i < n; ++i)
		{
			A->p[i + 1] = A->p[i] + nnz_per_col;
			std::vector<int> rand_vec;
			generate_uniq_rand_vector(nnz_per_col, n, rand_vec, seed);
			if (std::find(std::begin(rand_vec), std::end(rand_vec), i) ==
				std::end(rand_vec))
			{					 //if diagonal idx is not there,
				rand_vec[0] = i; //Put the diagonal in the first location
			}
			std::sort(rand_vec.begin(), rand_vec.end());
			for (int j = A->p[i], k = 0; j < A->p[i + 1]; ++j, ++k)
			{
				A->i[j] = rand_vec[k];
				A->x[j] = double_rand(0, max_val);
			}
		}
		return A;
	}

	CSC *random_symmetric_sparse(size_t n, double density, double max_val, unsigned seed)
	{
		// Warning: for large n, e.g., n>5e5, this is very slow
		int nnz_per_col = density * n + 1;
		nnz_per_col = nnz_per_col > n ? n : nnz_per_col;
		size_t nnz = n * nnz_per_col;
		if (nnz > INT32_MAX)
			return NULLPNTR;
		CSC *A = new CSC(n, n, nnz);
		// TODO
		return A;
	}

	bool is_equal(CSC *A, CSC *B, double eps)
	{
		if (A->n != B->n || A->nnz != B->nnz)
		{
			return false;
		}
		for (int j = 0; j < A->n + 1; ++j)
		{
			if (A->p[j] != B->p[j])
				return false;
		}
		for (int i = 0; i < A->nnz; ++i)
		{
			if (A->i[i] != B->i[i])
				return false;
		}
		for (int i = 0; i < A->nnz; ++i)
		{
			if (std::abs(A->x[i] - B->x[i]) > eps)
				return false;
		}
		return true;
	}

	void rhs_init(int n, int *Ap, int *Ai, double *Ax, double *b)
	{
		/*generating a rhs that produces a result of all 1 vector*/
		for (int j = 0; j < n; ++j)
		{
			b[j] = 0;
		}
		for (int c = 0; c < n; ++c)
		{
			for (int cc = Ap[c]; cc < Ap[c + 1]; ++cc)
			{
				b[Ai[cc]] += Ax[cc];
			}
		}
	}

	void rhs_init_blocked(size_t n, size_t nBlocks, size_t *Ap, int *Ai,
						  size_t *AiP, double *Ax, double *b)
	{
		/*generating a rhs that produces a result of all 1 vector*/
		for (int j = 0; j < n; ++j)
		{
			b[j] = 0;
		}
		for (int c = 0; c < n; ++c)
		{
			for (int cc = Ap[c], j = 0; cc < Ap[c + 1]; ++cc, ++j)
			{
				size_t curRow = Ai[AiP[c] + j];
				b[curRow] += Ax[cc];
			}
		}
	}

	bool test_unique(int n, int *vec)
	{
		auto *tmp = new bool[n]();
		for (int i = 0; i < n; ++i)
		{
			assert(vec[i] < n && vec[i] >= 0);
			tmp[vec[i]] = true;
		}
		for (int j = 0; j < n; ++j)
		{
			if (!tmp[j])
			{
				delete[] tmp;
				return false;
			}
		}
		delete[] tmp;
		return true;
	}



	 ///\Description: brief this function generates a blocked dataset that has a good load balance
	 // it also return the perfect schedule for this synthesize data
	 ///\param n number of columns
	 ///\param nnz number of non-zero elements inside the matrix
	 ///\param parts number of separated parts that each core should compute
	 ///\param cores number of processors to process this data
	 ///\param Ap the pointer array in CSC format
	 ///\param Ai the index array in CSC format
	 ///\param Ax the values inside the array with CSC format
	 ///\param perfect_schedule_ptr the pointer array in CSC format for nodes that going to be compute
	 ///\param perfect_schedule_set the index array in CSC format for nodes that going to be compute
	bool generateBlockedData(int n, int nnz, int part_per_core, int cores,
                             std::vector<int>& Ap, std::vector<int>& Ai, std::vector<double>& Ax,
                             std::vector<int>& perfect_schedule_ptr, std::vector<int>& perfect_schedule_set, bool apply_max_nnz){
        assert(cores != 0);
        assert(part_per_core != 0);
        assert(n <= nnz);
        assert(nnz > 0);
        int parts = part_per_core * cores;
	    if(n < parts){
	        std::cerr << "There is not enough nodes to be divided between cores!\t" << n << "<" << parts << std::endl;
            return false;
	    }

	    //Computing columns per block
        int column_per_block = ceil((double) n / parts);
        int column_last_block = -1;
	    while(column_last_block < 0){
            column_last_block = n - column_per_block * (parts - 1);
            if(column_last_block < 0){
                column_per_block--;
            }
	    }
	    if(column_last_block < 0){
	        std::cerr << "There is an error in the generateBlockedData function for column_last_block" << std::endl;
	    }
	    //Computing the maximum possible nnz
	    long long unsigned int max_block_size = 0.5 * (column_per_block * column_per_block - column_per_block) + column_per_block;
	    long long unsigned int max_nnz = max_block_size * parts;
	    if(apply_max_nnz){
	        nnz = max_nnz;
	    } else if(max_nnz < nnz && max_nnz > 0){
	        std::cerr << "The number of nnz requested is more than the capability of this function" << std::endl;
	        std::cerr << "Requested nnz is:" << nnz << "\t" << "Maximum nnz is:" << max_nnz << std::endl;
            return false;
	    } else {
	        std::cout << "The number of nnz requested is OK to generate perfect data" << std::endl;
	    }
	    //Computing the nnz per block
	    int nnz_with_no_diag = nnz - n;
        int nnz_per_block = ceil( (double) nnz_with_no_diag / parts);
	    int nnz_last_block = nnz_with_no_diag - nnz_per_block * (parts - 1);
	    if(nnz_last_block < 0){
	        std::cerr << "There is an error in the generateBlockedData function for nnz_last_block" << std::endl;
	        return false;
	    }
        //Creating blocks
        Ap.resize(n + 1);
	    Ai.resize(nnz);
	    Ax.resize(nnz);
        Ap[0] = 0;
        int last_pointer = 0;
        for(int core = 0; core < parts; core++){
            int nnz_left = 0;
            int current_num_columns = 0;
            if(core != parts - 1){
                nnz_left = nnz_per_block;
                current_num_columns = column_per_block;
            } else {
                nnz_left = nnz_last_block;
                current_num_columns = column_last_block;
            }
            std::vector<std::vector<int>> index(current_num_columns, std::vector<int>(1,0));
            //Initializing diag index per block
            int first_column = core * column_per_block;
            int tmp = first_column + current_num_columns;
            int last_column = std::min(tmp, n);
            for(int diag = first_column; diag < last_column; diag++){
                index[diag - first_column][0] = diag;
            }
            //Adding nnz in a round rubin form

            // The last column already has its diag element
            last_column -= 2;

            while( nnz_left > 0){
                for(int column = last_column; column >= first_column; column--){
                    index[column - first_column].push_back(index[column - first_column].back() + 1);
                    nnz_left--;
                    if(nnz_left == 0){
                        break;
                    }
                }
                last_column--;
            }
            //Adding the block to the matrix
            for(auto& row: index){
                Ap[last_pointer + 1] = Ap[last_pointer] + row.size();
                std::copy(row.begin(), row.end(), Ai.data() + Ap[last_pointer]);
                last_pointer++;
            }
        }

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(1.0, 10.0);
        //Init Ax with random number
        for(int i = 0; i < nnz; i++){
            Ax[i] = dist(mt);
        }
        //Probably this makes it SPD
        for(int i = 0; i < n; i++){
            Ax[Ap[i]] = 10000;
        }
        assert(Ap[n] == nnz);

        //Computing Perfect Schedule
        perfect_schedule_ptr.resize(cores + 1, 0);
        perfect_schedule_set.resize(n, 0);
        perfect_schedule_ptr[0] = 0;
        for(int i = 0; i < cores; i++){
            int node_per_parallel_block = 0;
            if(i == cores - 1){
                node_per_parallel_block = (part_per_core - 1) * column_per_block + column_last_block;
            } else {
                node_per_parallel_block = part_per_core * column_per_block;
            }
            perfect_schedule_ptr[i + 1] = perfect_schedule_ptr[i] + node_per_parallel_block;
        }
        for(int node = 0; node < n; node++){
            perfect_schedule_set[node] = node;
        }
        assert(perfect_schedule_ptr[cores] == n);
        assert(Ap[n] == nnz);
        return true;
	}


	CSC* generateDAGfromEdgeList(int num_nodes, std::vector<std::vector<bool>>& DAG){
        if(num_nodes != DAG.size()){
            std::cerr << "The DAG initial dimension should have equal size with num_nodes "
                << num_nodes << "!=" << DAG.size() << std::endl;
            return NULLPNTR;
        }

        int nnz = 0;
        //If there is no diagonal edge, add them and then count the number of edges
        for(int node = 0; node < num_nodes; node++){
            if(!DAG[node][node]){
                DAG[node][node] = true;
            }
            for(int i = 0; i < num_nodes; i++){
                if(DAG[node][i]){
                    nnz++;
                }
            }
        }

        CSC *A = new CSC(num_nodes, num_nodes, nnz);

        auto DAG_ptr = A->p;
        auto DAG_set = A->i;
        auto DAG_x = A->x;
        int node, edge_cnt;

        for(node = 0, edge_cnt = 0; node < num_nodes; node++){
            DAG_ptr[node] = edge_cnt;
            for(int col = 0; col < num_nodes; col++) {
                if(DAG[node][col]){
                    DAG_set[edge_cnt++] = col;
                }
            }
        }
        DAG_ptr[num_nodes] = edge_cnt;

        for(int i = 0; i < edge_cnt; i++){
            DAG_x[i] = double_rand(10,20);
        }

        return A;
	}

	bool addEdge(std::vector<std::vector<bool>>& DAG, int i, int j, bool showWarning){
	    if(DAG.size() <= i && DAG[i].size() <= j){
	        std::cerr << "The size of the vector is not appropriate for adding i and j,"
                         " please resize the vector." << std::endl;
	        return false;
	    }

	    if(DAG[i][j] && showWarning){
	        std::cerr << "The edge is already added." << std::endl;
	    }

	    DAG[i][j] = true;
	    return true;
	}



} // namespace sym_lib
