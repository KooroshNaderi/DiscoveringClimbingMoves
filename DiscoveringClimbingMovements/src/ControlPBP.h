/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/


#ifndef CONTROLPBP_H
#define CONTROLPBP_H
#include <Eigen/Eigen> 
#include <vector>
#include "DiagonalGMM.h"
#include "DynamicPdfSampler.h"
#include "TrajectoryOptimization.h"

#ifdef SWIG
#define __stdcall  //SWIG doesn't understand __stdcall, but fortunately c# assumes it for virtual calls
#endif


namespace AaltoGames
{
	//Defines how controls affect system state at each step 
	typedef void (*TransitionFunction)(int step, const float *state, const float *controls, float *out_state);
	//Evaluates the cost for a state, e.g., penalizes not hitting a target. Returned in the form of squared cost, which can be transformed into a potential as exp(-sqCost) by probabilistic optimizers.
	typedef double (*StateCostFunction)(int stepIdx, const float *state);
	//Optional notification callback after the update algorithm has processed a single step (node) of the graph
	typedef void (*OnStepDone)();

	//Implements the C-PBP algorithm as in the Hämäläinen et al. 2015 paper "Online Control of Simulated Humanoids Using Particle Belief Propagation"
	//Note: although we use Eigen vectors internally, the public interface uses only plain float arrays for maximal portability
	class ControlPBP : public ITrajectoryOptimization
	{
	public:
		ControlPBP();
		~ControlPBP();
		//minValues and maxValues contain first the bounds for state variables, then for control variables
		//stateKernelStd==NULL corresponds to the special case of Q=0
		//Note that instead of specifying the Q and sigmas of the paper, the values are provided as float standard deviations corresponding to the diagonal elements, 
		//controlPriorStd=sqrt(diag. of \sigma_{0}^2 C_u), controlPriorDiffStd = sqrt(diag. of \sigma_{1}^2 C_u), controlPriorDiffDiffStd = sqrt(diag. of \sigma_{2}^2 C_u)
		void init(int nParticles, int nSteps, int nStateDimensions, int nControlDimensions, const float *controlMinValues, const float *controlMaxValues, const float *controlMean, const float *controlPriorStd, const float *controlDiffPriorStd, const float *controlDiffDiffPriorStd, float controlMutationStdScale, const float *stateKernelStd);
		//Note that this is just a convenience method that internally calls startIteration(), getControl() etc.
		virtual void __stdcall update(const float *currentState, TransitionFunction transitionFwd, StateCostFunction statePotential, OnStepDone onStepDone=NULL);
		virtual double __stdcall getBestTrajectoryCost();
		virtual void __stdcall getBestControl(int timeStep, float *out_control);
		virtual void __stdcall getBestControlState( int timeStep, float *out_state );
		//Returns the original state cost for the best trajectory passed from the client to ControlPBP for the given timestep. This is mainly for debug and testing purposes.
		virtual double __stdcall getBestTrajectoryOriginalStateCost( int timeStep);
		virtual void __stdcall setSamplingParams(const float *controlPriorStd, const float *controlDiffPriorStd, const float *controlDiffDiffPriorStd, float controlMutationStdScale, const float *stateKernelStd);

		//returns the prior GMM for the given time step
#ifndef SWIG
		void getConditionalControlGMM(int timeStep, const Eigen::VectorXf &state, DiagonalGMM &dst);
#endif
		/*
		Below, an interface for operation without callbacks. This is convenient for C# integration and custom multithreading. See InvPendulumTest.cpp for the usage.
		*/
		virtual void __stdcall startIteration(bool advanceTime, const float *initialState);
		virtual void __stdcall startPlanningStep(int stepIdx);
		//typically, this just returns sampleIdx. However, if there's been a resampling operation, multiple new samples may link to the same previous sample (and corresponding state)
		virtual int __stdcall getPreviousSampleIdx(int sampleIdx);
		//samples a new control, considering an optional gaussian prior with diagonal covariance (this corresponds to the \mu_p, \sigma_p, C_u in the paper, although the C_u is computed on Unity side and raw stdev and mean arrays are passed to the optimizer)
		virtual void __stdcall getControl(int sampleIdx, float *out_control, const float *priorMean=0, const float *priorStd=0);
		virtual void __stdcall updateResults(int sampleIdx, const float *finalControl, const float *newState, double stateCost, const float *priorMean=0, const float *priorStd=0, float extControlCost=-1);
		virtual void __stdcall endPlanningStep(int stepIdx);
		virtual void __stdcall endIteration();
		//uniformBias: portion of no-prior samples in the paper (0..1)
		//resampleThreshold: resampling threshold, same as in the paper. Default 0.5
		//useGaussianBackPropagation: if true, the Gaussian local refinement (Algorithm 2 of the paper) is used. 
		//gbpRegularization: the regularization of Algorithm 2. Default 0.001 
		virtual void __stdcall setParams(double uniformBias, double resampleThreshold, bool useGaussianBackPropagation,float gbpRegularization);
		//this function is public only for debug drawing
		void gaussianBackPropagation(TransitionFunction transitionFwd=NULL, OnStepDone onStepDone=NULL);
		virtual int __stdcall getBestSampleLastIdx();
		float getGBPRegularization();
		//Eigen::MatrixXf marginalDataToMatrix(void);
		//Eigen::MatrixXf ControlPBP::stateAndControlToMatrix(void);

		
		//A vector of MarginalSamples for each graph node, representing a GMM of joint state and control 
		//This is made public for easier debug visualizations, but you should treat this as read-only data.
#ifndef SWIG
		std::vector<std::vector<MarginalSample> > marginals;
#endif
		void reset()
		{
			iterationIdx=0;
		}
	private:
		virtual void __stdcall getControlWithoutStateKernel(int sampleIdx, float *out_control, const float *priorMean=0, const float *priorStd=0);
		//The transitions in a matrix. Each row has one transition. The first nStateDimensions columns have the previous state, the next nStateDimensions have the current state, the next nControlDimensions have the controls of the transition.
		Eigen::MatrixXf transitionData;



		std::vector<Eigen::VectorXf> fullSamples;
		std::vector<MarginalSample> oldBest;
		std::vector<double> fullSampleCosts;
		double bestCost;
		Eigen::VectorXf gaussianBackPropagated;
		std::vector<DiagonalGMM> prior;
		Eigen::VectorXf controlMin,controlMax,controlMean, controlPriorStd,controlDiffPriorStd,controlDiffDiffPriorStd,controlMutationStd,stateKernelStd;
		int nSteps;
		DiagonalGMM staticPrior;
		int nSamples;
		int nStateDimensions;
		int nControlDimensions;
		int iterationIdx;
		bool resample;
		int currentStep;
		int nextStep;
		int nInitialGuesses;
		bool useStateKernels;
		int bestFullSampleIdx;
		bool timeAdvanced;

		DynamicPdfSampler *selector;
		DynamicPdfSampler *conditionalSelector;
		void finalizeWeights(int stepIdx);
		double uniformBias;
		double resampleThreshold;  //0 = never resample, >1 = always resample
		bool useGaussianBackPropagation;
		float gbpRegularization;

	};

} //namespace AaltoGames


#endif //CONTROLPBP_H


