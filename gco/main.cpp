
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "src/GCoptimization.h"


void GeneralGraph_DArraySArraySpatVarying(int num_pixels, int num_labels, int num_edges, int *probabilites, int *edges)
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

		printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion();
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;

}
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
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

	int *probabilities = new int[num_pixels*num_labels];

	std::ifstream infile2("nodes.txt");
	int a,b,c,d,e,f ;
	while (infile2 >> a >> b >> c >> d >> e >> f)
	{
		probabilities[num_pixels*6] = -a;
		probabilities[num_pixels*6+1] = -b;
		probabilities[num_pixels*6+2] = -c;
		probabilities[num_pixels*6+3] = -d;
		probabilities[num_pixels*6+4] = -e;
		probabilities[num_pixels*6+5] = -f;
		num_pixels++;
	}

	//get edges
	std::ifstream infile3("edges.txt");
	int a1,b1 ;
	while (infile3 >> a1 >> b1)
	{
		num_edges++;
	}

	int *edges = new int[num_edges*2];

	std::ifstream infile4("edges.txt");
	int a2,b2 ;
	while (infile4 >> a2 >> b2)
	{
		edges[num_edges*2] = a2;
		edges[num_edges*2+1] = b2;
		num_edges++;
	}
	//Will pretend our graph is general, and set up a neighborhood system
	// which actually is a grid. Also uses spatially varying terms
	GeneralGraph_DArraySArraySpatVarying(num_pixels,num_labels,num_edges,probabilities,edges);

	printf("\n  Finished %d (%d) clock per sec %d",clock()/CLOCKS_PER_SEC,clock(),CLOCKS_PER_SEC);

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////

