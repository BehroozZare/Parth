#include "equalize_valences.h"
#include "collapse_edges.h"
#include "tangential_relaxation.h"
#include <igl/is_edge_manifold.h>
#include <igl/writeOBJ.h>
#include "split_edges_until_bound.h"
#include <igl/unique_edge_map.h>
#include <igl/edge_flaps.h>
#include <igl/circulation.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/avg_edge_length.h>
#include <iostream>


void remesh_botsch_map(Eigen::MatrixXd & V,Eigen::MatrixXi & F, std::vector<int> & old_to_new_dof_map,
                   Eigen::VectorXd & target,int iters,
                   Eigen::VectorXi & feature, bool project){
    Eigen::MatrixXd V0;
    Eigen::MatrixXi F0;

    Eigen::VectorXd high,low,lambda;
    high = 1.4*target;
    low = 0.7*target;

    F0 = F;
    V0 = V;

    old_to_new_dof_map.resize(V.rows());
    //Initialize old_to_new vectors to identity vector
    for (int i = 0; i < V.rows(); i++) {
        old_to_new_dof_map[i] = i;
    }

    // Iterate the four steps
    for (int i = 0; i<iters; i++) {
        //=============== Step 1: Split ===============
        // New vertices are added to the end of V
        split_edges_until_bound(V,F,feature,high,low);
        //=============== Step 2: Collapse ===============
        Eigen::VectorXi I; // indices to the birth vertices
        int prev_num_vertices = V.rows();
        collapse_edges(V,F,feature,high,low, I); // Collapse
        //--------- compute the map per iteration
        //Create the mapping vector between old and new dofs ids
        assert(I.rows() == V.rows());
        std::vector<int> old_to_new_iter_dof_map(prev_num_vertices, -1);
        for(int j = 0; j < I.rows(); j++){
            old_to_new_iter_dof_map[I(j)] = j;
        }
        //--------- compute the end-to-end map
        for(int j = 0; j < old_to_new_dof_map.size(); j++){
            if(old_to_new_dof_map[j] != -1){
                old_to_new_dof_map[j] = old_to_new_iter_dof_map[old_to_new_dof_map[j]];
            }
        }
        //=============== Step 3: Flip ===============
        equalize_valences(V,F,feature); // Flip
        int n = V.rows();
        lambda = Eigen::VectorXd::Constant(n,1.0);
        if(!project){
            V0 = V;
            F0 = F;
        }
        //=============== Step 4: Relax ===============
        tangential_relaxation(V,F,feature,V0,F0,lambda); // Relax
    }
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F,
                   Eigen::VectorXd & target,int iters,
                   Eigen::VectorXi & feature, bool project){
    Eigen::MatrixXd V0;
    Eigen::MatrixXi F0;

    Eigen::VectorXd high,low,lambda;
    high = 1.4*target;
    low = 0.7*target;

	F0 = F;
	V0 = V;

    // Iterate the four steps
    for (int i = 0; i<iters; i++) {
        //=============== Step 1: Split ===============
        // New vertices are added to the end of V
    	split_edges_until_bound(V,F,feature,high,low);
        //=============== Step 2: Collapse ===============
        Eigen::VectorXi I; // indices to the birth vertices
    	collapse_edges(V,F,feature,high,low, I); // Collapse
        //=============== Step 3: Flip ===============
    	equalize_valences(V,F,feature); // Flip
    	int n = V.rows();
    	lambda = Eigen::VectorXd::Constant(n,1.0);
	if(!project){
		V0 = V;
		F0 = F;
	}
        //=============== Step 4: Relax ===============
	tangential_relaxation(V,F,feature,V0,F0,lambda); // Relax
    }
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters, Eigen::VectorXi & feature){
remesh_botsch(V,F,target,iters,feature,false);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters){
	Eigen::VectorXi feature;
	feature.resize(0);
	remesh_botsch(V,F,target,iters,feature);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target,int iters, bool project){
	Eigen::VectorXi feature;
	feature.resize(0);
	remesh_botsch(V,F,target,iters,feature,project);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, Eigen::VectorXd & target){
	int iters = 10;
	remesh_botsch(V,F,target,iters);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters){
	Eigen::VectorXi feature;
	feature.resize(0);
	Eigen::VectorXd target;
	int n = V.rows();
	target = Eigen::VectorXd::Constant(n,target_double);
	remesh_botsch(V,F,target,iters,feature);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters, bool project){
	Eigen::VectorXi feature;
	feature.resize(0);
	Eigen::VectorXd target;
	int n = V.rows();
	target = Eigen::VectorXd::Constant(n,target_double);
	remesh_botsch(V,F,target,iters,feature,project);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double,int iters, Eigen::VectorXi feature, bool project){
	Eigen::VectorXd target;
	int n = V.rows();
	target = Eigen::VectorXd::Constant(n,target_double);
	remesh_botsch(V,F,target,iters,feature,project);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F, double target_double){
	int iters = 10;
	Eigen::VectorXd target;
	int n = V.rows();
	target = Eigen::VectorXd::Constant(n,target_double);
	remesh_botsch(V,F,target,iters);
}

void remesh_botsch(Eigen::MatrixXd & V,Eigen::MatrixXi & F){
	double h = igl::avg_edge_length(V,F);
	Eigen::VectorXd target;
	int n = V.rows();
	target = Eigen::VectorXd::Constant(n,h);
	remesh_botsch(V,F,target);
}
// g++ -I/usr/local/libigl/external/eigen -I/usr/local/libigl/include -std=c++11 -framework Accelerate main.cpp remesh_botsch.cpp -o main

