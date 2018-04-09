/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/

#include "DynamicMinMax.h"
#include "Debug.h"

namespace AaltoGames
{

	void DynamicMinMax::init( int N, DynamicMinMax *parent/*=NULL*/ )
	{
		children[0]=0;
		children[1]=0;
		this->parent=parent;
		root=this;
		hasChildren=false;
		while (root->parent!=NULL){
			root=root->parent;
		}
		if (N>1)
		{
			hasChildren=true;
			//divide until we have a child for each discrete pdf element. 
			//also gather the subtree leaves to the leaves vector
			int NChildren[2]={N/2,N-N/2};
			for (int k=0; k<2; k++)
			{
				children[k]=new DynamicMinMax();
				children[k]->init(NChildren[k],this);
				for (size_t i=0; i<children[k]->leaves.size(); i++)
				{
					leaves.push_back(children[k]->leaves[i]);
				}
			}

		}
		else{
			leaves.push_back(this);
		}
		//at root, update the pdf element (bin) indices of all leaves
		if (parent==NULL)
		{
			for (size_t i=0; i<leaves.size(); i++)
			{
				leaves[i]->elemIdx=(int)i;
				leaves[i]->minChildIdx=(int)i;
				leaves[i]->maxChildIdx=(int)i;
			}
		}

		//init minChildIdx and maxChildIdx to 0 (doesn't matter what, just something that will not point outside the allocated data)
		minChildIdx=0;
		maxChildIdx=0;
	}

	DynamicMinMax::DynamicMinMax()
	{
		hasChildren = false;
	}

	DynamicMinMax::~DynamicMinMax()
	{
		if (hasChildren)
		{
			for (int k=0; k<2; k++)
			{
				delete children[k];
			}
		}
	}

	void DynamicMinMax::setValue( int idx, double value )
	{
		leaves[idx]->value=value;
		DynamicMinMax *p=leaves[idx];
		while (p->parent!=NULL)
		{
			p=p->parent;
			if (p->hasChildren)
			{
				AALTO_ASSERT1(p->children[0]!=NULL && p->children[1]!=NULL);
				p->minChildIdx = leaves[p->children[0]->minChildIdx]->value < leaves[p->children[1]->minChildIdx]->value
					? p->children[0]->minChildIdx : p->children[1]->minChildIdx; 
				p->maxChildIdx = leaves[p->children[0]->maxChildIdx]->value > leaves[p->children[1]->maxChildIdx]->value
					? p->children[0]->maxChildIdx : p->children[1]->maxChildIdx; 
			}
		}
	}

	double DynamicMinMax::getValue( int idx )
	{
		return leaves[idx]->value;
	}

	int DynamicMinMax::getMinIdx()
	{
		return minChildIdx;
	}
	int DynamicMinMax::getMaxIdx()
	{
		return maxChildIdx;
	}

}