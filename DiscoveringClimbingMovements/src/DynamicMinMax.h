/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/


#ifndef DynamicMinMax_H
#define DynamicMinMax_H 
#include <vector>
#include "MathUtils.h"

namespace AaltoGames
{

	///Sampler for sampling from a discrete and dynamically changing pdf.
	///This is implemented using a tree structure where the discrete probabilities are propagated towards the root
	class DynamicMinMax
	{
	public:
		void init(int N, DynamicMinMax *parent=NULL);
		~DynamicMinMax();
		DynamicMinMax();
		void setValue(int idx, double value);
		double getValue(int idx);
		int getMinIdx();
		int getMaxIdx();
	protected:
		DynamicMinMax *children[2];
		DynamicMinMax *parent;
		DynamicMinMax *root;
		bool hasChildren;
		int elemIdx;
		int minChildIdx,maxChildIdx;
		double value;
		std::vector<DynamicMinMax *> leaves;
	};

}

#endif