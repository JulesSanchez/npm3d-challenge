
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "src/GCoptimization.h"


int* AlphaExpansionOnApproximateGraph(int num_pixels, int num_labels, int num_edges, int *probabilites, int *edges)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = l1 == l2  ? 0:100;


	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(probabilites);
		gc->setSmoothCost(smooth);

		// now set up a grid neighborhood system
		// first set up horizontal neighbors
		for (int y = 0; y < num_edges; y++ ){
			int p1 = edges[2*y];
			int p2 = edges[2*y+1];
			gc->setNeighbors(p1,p2);
			}

		printf("\nBefore optimization energy is %lld", gc->compute_energy());
		gc->expansion();
		printf("\nAfter optimization energy is %lld", gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] smooth;
	return result;

}
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	printf("\nOn est tipar");
	long num_pixels = 0;
	int num_labels = 6;
	long num_edges = 0;

	//get nodes
	std::ifstream infile1("nodes.txt");
	int a_,b_,c_,d_,e_,f_;
	while (infile1 >> a_ >> b_ >> c_ >> d_ >> e_ >> f_)
	{
		num_pixels++;
	}
	printf("\nNodes lu 1st time");
	int *probabilities = new int[num_pixels*num_labels];

	std::ifstream infile2("nodes.txt");
	int a,b,c,d,e,f ;
	long new_pixels = 0;
	while (infile2 >> a >> b >> c >> d >> e >> f)
	{
		probabilities[new_pixels*6] = -a;
		probabilities[new_pixels*6+1] = -b;
		probabilities[new_pixels*6+2] = -c;
		probabilities[new_pixels*6+3] = -d;
		probabilities[new_pixels*6+4] = -e;
		probabilities[new_pixels*6+5] = -f;
		new_pixels++;
	}
	printf("\nNodes lu 2nd time");

	//get edges
	std::ifstream infile3("edges.txt");
	int a1,b1 ;
	while (infile3 >> a1 >> b1)
	{
		num_edges++;
	}

	printf("\nEdges lu 1st time");
	int *edges = new int[num_edges*2];
	long new_edges = 0;

	std::ifstream infile4("edges.txt");
	int a2,b2 ;
	while (infile4 >> a2 >> b2)
	{
		edges[new_edges*2] = a2;
		edges[new_edges*2+1] = b2;
		new_edges++;
	}
	printf("\nEdges lu 2nd time");
	//Will pretend our graph is general, and set up a neighborhood system
	// which actually is a grid. Also uses spatially varying terms
	int* result = AlphaExpansionOnApproximateGraph(num_pixels,num_labels,num_edges,probabilities,edges);
	std::ofstream o("labels.txt");
	for(int i = 0; i<num_pixels;i++){
		o<<result[i]+1<<"\n";
	}
	printf("\n  Finished %lld (%lld) clock per sec %lld",clock()/CLOCKS_PER_SEC,clock(),CLOCKS_PER_SEC);

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////

