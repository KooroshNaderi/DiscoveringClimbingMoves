#include <stdio.h>
#include <conio.h>
#include <chrono>
#include <list>
#include <unordered_map>
#include <queue>
#include <stdint.h>
#include "ode/ode.h"
#include "MathUtils.h"
#include <Math.h>

#include "ControlPBP.h"
#include "UnityOde.h"
#include "RenderClient.h"
#include "RenderCommands.h"
//#include "src\Debug.h"
#include "DynamicMinMax.h"
#include "CMAES.hpp"
#include "ClippedGaussianSampling.h"
#include "RecursiveTCBSpline.h"
#include "FileUtils.h"

using namespace std::chrono;
using namespace std;
using namespace AaltoGames;
using namespace Eigen;

#ifdef _MSC_VER
#pragma warning(disable:4244 4305)  /* for VC++, no precision loss complaints */
#endif

/// select correct drawing functions 
#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#endif
enum OptimizerTypes
{
	otCPBP=0,otCMAES
};
static const OptimizerTypes optimizerType = otCMAES;

float xyz[3] = {4.0,-5.0f,0.5f}; 
float lightXYZ[3]={-10,-10,10};
float lookAt[3]={0.75f,0,2.0};


static bool pause = false;
static const bool useOfflinePlanning = true;
static const bool flag_capture_video = false;
static int maxNumSimulatedPaths = 10;

enum mCaseStudy{movingLimbs = 0, Energy = 1};
//case study: {path with minimum moving limbs, path with minimum energy} among "maxNumSimulatedPaths" paths founds
static int maxCaseStudySimulation = 2; 

static float mBiasWallY = -0.2f;
static float maxSpeed = optimizerType==otCMAES ? 2.0f*3.1415f : 2.0f*3.1415f;
static const float poseAngleSd=deg2rad*45.0f;
static const float maxSpeedRelToRange=4.0f; //joint with PI rotation range will at max rotate at this times PI radians per second
static const float controlDiffSdScale =useOfflinePlanning ? 0.5f : 0.2f;
// for online method it works best if nPhysicsPerStep == 1. Also, Putting nPhysicsPerStep to 3 does not help in online method and makes simulation awful (bad posture, not reaching a stance)
static int nPhysicsPerStep = optimizerType==otCMAES ? 1 : (useOfflinePlanning ? 4 : 1);


static const float noCostImprovementThreshold = optimizerType == otCMAES ? 50.0f : 50.0f;
static float cTime = 30.0f;
static float contactCFM = 0; 

//CMAES params
static const int nCMAESSegments=2;
static const bool cmaesLinearInterpolation=true;
static const bool forceCMAESLastValueToZero=false;
static const float minSegmentDuration=0.25f;
static const float maxSegmentDuration=0.75f;
static const float maxFullTrajectoryDuration=3.0f;
static const float motorPoseInterpolationTime=0.0333f;
static const float torsoMinFMax=20.0f;
static const bool optimizeFMax=false;
static const bool cmaesSamplePoses=false;
static const bool useThreads=true;

enum FMaxGroups
{
	fmTorso=0,fmLeftArm ,fmLeftLeg,fmRightArm,fmRightLeg,
	fmEnd
};
static const int fmCount=optimizeFMax ? fmEnd : 0;


int nRows = 3;
int nColumns = 3;
int nTestNum = 10;
//static int numRunForHold = 3;
//static int numHoldsTest = 10;
//static bool isDynamicGraph = true;
//static float trialTimeBeforeChangePath = 100;

//Simulation globals
static float worldCfM = 1e-3;
static float worldERP = 1e-5;
static const float maximumForce = 250.0f;//250.0f (strong) or 50.0f (weak)
static const float minimumForce = 2.0f; //non-zero FMax better for joint stability
static const float forceCostSd=optimizerType==otCMAES ? maximumForce : maximumForce;  //for computing control cost

static const bool testClimber = false;
// initial rotation (for cost computing in t-pose)
static	__declspec(align(16)) Eigen::Quaternionf initialRotations[100];  //static because the class of which this should be a member should be modified to support alignment...


 /// some constants 
static const float timeStep = 1.0f / cTime;   //physics simulation time step
#define BodyNUM 11+4		       // number of boxes 
//Note: CMAES uses this only for the first iteration, next ones use 1/4th of the samples
static const int contextNUM = optimizerType==otCMAES ? 257 : 65; //maximum number of contexts, //we're using 32 samples, i.e., forward-simulated trajectories per animation frame.
#define boneRadius (0.2f)	   // sphere radius 
#define boneLength (0.5f)	   // sphere radius
#define DENSITY 1.0f		// density of all objects
#define holdSize 0.5f

static const int nTrajectories = contextNUM - 1;
static int nTimeSteps = useOfflinePlanning ? int(cTime*1.5001f) : int(cTime/2);

#define Vector2 Eigen::Vector2f
#define Vector3 Eigen::Vector3f
#define Vector4 Eigen::Vector4f

void start(); /// start simulation - set viewpoint 
void simLoop (int); /// simulation loop
void runSimulation(int, char **); /// run simulation

//WorkerThreadManager<int> *workerThreadManager;

enum mEnumTestCaseClimber
{
	TestAngle = 0, TestCntroller = 1
};

enum mDemoTestClimber{DemoRoute1 = 1, DemoRoute2 = 2, DemoRoute3 = 3, DemoLongWall = 4, Demo45Wall = 5, Line = 6, DemoLongWallTest = 7, Demo45Wall2 = 8};

class CMAESTrajectoryResults
{
public:
	static const int maxTimeSteps=512;
	VectorXf control[maxTimeSteps];
	int nSteps;
	float cost;
};

static inline Eigen::Quaternionf ode2eigenq(ConstOdeQuaternion odeq)
{
	return Eigen::Quaternionf(odeq[0],odeq[1],odeq[2],odeq[3]); 
}
static inline void eigen2odeq(const Eigen::Quaternionf &q, OdeQuaternion out_odeq)
{
	out_odeq[0]=q.w();
	out_odeq[1]=q.x();
	out_odeq[2]=q.y();
	out_odeq[3]=q.z();
}

//timing
high_resolution_clock::time_point t1;
void startPerfCount()
{
	t1 = high_resolution_clock::now();
}
int getDurationMs()
{
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	return (int)(time_span.count()*1000.0);
}

//the full state of a body
class BodyState
{
public:
	BodyState()
	{
		setAngle(Vector4(0.0f,0.0f,0.0f,0.0f));
		setPos(Vector3(0.0f,0.0f,0.0f));
		setVel(Vector3(0.0f,0.0f,0.0f));
		setAVel(Vector3(0.0f,0.0f,0.0f));
		boneSize = boneLength;
		bodyType = 0;
	}

	Vector4 getAngle()
	{
		return Vector4(angle[0], angle[1], angle[2], angle[3]);
	}

	Vector3 getPos()
	{
		return Vector3(pos[0], pos[1], pos[2]);
	}

	Vector3 getVel()
	{
		return Vector3(vel[0], vel[1], vel[2]);
	}

	Vector3 getAVel()
	{
		return Vector3(aVel[0], aVel[1], aVel[2]);
	}

	float getBoneSize()
	{
		return boneSize;
	}

	int getBodyType()
	{
		return bodyType;
	}

	float getBodyMass()
	{
		return boneMass;
	}

	void setAngle(Vector4& iAngle)
	{
		angle[0] = iAngle[0];
		angle[1] = iAngle[1];
		angle[2] = iAngle[2];
		angle[3] = iAngle[3];
		return;
	}

	void setPos(Vector3& iPos)
	{
		pos[0] = iPos[0];
		pos[1] = iPos[1];
		pos[2] = iPos[2];
		return;
	}

	void setVel(Vector3& iVel)
	{
		vel[0] = iVel[0];
		vel[1] = iVel[1];
		vel[2] = iVel[2];
		return;
	}

	void setAVel(Vector3& iAVel)
	{
		aVel[0] = iAVel[0];
		aVel[1] = iAVel[1];
		aVel[2] = iAVel[2];
		return;
	}

	void setBoneSize(float iBonSize)
	{
		boneSize = iBonSize;
	}

	void setBodyType(float iBodyType)
	{
		bodyType = iBodyType;
	}
	
	void setBodyMass(float iBodyMass)
	{
		boneMass = iBodyMass;
	}

private:
	float angle[4];
	float pos[3];
	float vel[3];
	float aVel[3];

	float boneSize;
	int bodyType;
	float boneMass;
};

class BipedState
{
public:
	enum BodyName{ BodyTrunk = 0, BodyLeftThigh = 1, BodyRightThigh = 2, BodyLeftShoulder = 3, BodyRightShoulder = 4
				 , BodyLeftLeg = 5, BodyRightLeg = 6, BodyLeftArm = 7, BodyRightArm = 8, BodyHead = 9, BodySpine = 10
				 , BodyLeftHand = 11, BodyRightHand = 12, BodyLeftFoot = 13, BodyRightFoot = 14
				 , BodyTrunkUpper = 15, BodyTrunkLower = 16, BodyHold = 17 };
	enum BodyType{BodyCapsule = 0, BodyBox = 1, BodySphere = 2};

	BipedState()
	{
		for (int i = 0; i < BodyNUM; i++)
		{
			bodyStates.push_back(BodyState());
		}
	}

	BipedState getNewCopy(int freeSlot, int fromContextSlot)
	{
		BipedState c;

		c.toDesAngles = toDesAngles;

		c.bodyStates = bodyStates;
		c.hold_bodies_ids = hold_bodies_ids;
		c.saving_slot_state = freeSlot;
		saveOdeState(freeSlot, fromContextSlot);

		return c;
	}

	std::vector<float> toDesAngles;

	std::vector<BodyState> bodyStates;
	int saving_slot_state;
	std::vector<int> hold_bodies_ids; // left-leg, right-get, left-hand, right-hand

	Vector3 computeCOM()
	{
		float totalMass=0;
		Vector3 result=Vector3::Zero();
		for (int i = 0; i < BodyNUM; i++)
		{
			float mass = bodyStates[i].getBodyMass(); // odeBodyGetMass(bodyIDs[i]);
			Vector3 pos = bodyStates[i].getPos(); // (odeBodyGetPosition(bodyIDs[i]));
			result+=mass*pos;
			totalMass+=mass;
		}
		return result/totalMass;
	}

	Vector3 getBodyDirectionZ(int i)
	{
		dMatrix3 R;
		Vector4 mAngle;
		if (i == BodyName::BodyTrunkLower)
			mAngle = bodyStates[BodyName::BodySpine].getAngle();
		else
			mAngle = bodyStates[i].getAngle();
		dReal mQ[]= {mAngle[0], mAngle[1], mAngle[2], mAngle[3]};
		dRfromQ(R, mQ);

		int targetDirection = 2; // alwayse in z direction
		int t_i = i;
		if (i == BodyName::BodyTrunkLower)
			t_i = BodyName::BodySpine;
		if (bodyStates[t_i].getBodyType() == BodyType::BodyBox)
		{
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder)
				targetDirection = 0; // unless one of these bones is wanted
		}
		Vector3 targetDirVector(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
		if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyLeftThigh || i == BodyName::BodyRightThigh 
			|| i == BodyName::BodyLeftLeg || i == BodyName::BodyRightLeg || i == BodyName::BodyTrunkLower || i == BodyName::BodyLeftHand
			|| i == BodyName::BodyLeftFoot || i == BodyName::BodyRightFoot)
		{
			targetDirVector = -targetDirVector;
		}

		return targetDirVector;
	}

	Vector3 getEndPointPosBones(int i)
	{
		/*switch (i)
		{
			case BodyName::BodyLeftArm:
				return bodyStates[BodyName::BodyLeftHand].getPos();
				break;
			case BodyName::BodyLeftLeg:
				return bodyStates[BodyName::BodyLeftFoot].getPos();
				break;
			case BodyName::BodyRightArm:
				return bodyStates[BodyName::BodyRightHand].getPos();
				break;
			case BodyName::BodyRightLeg:
				return bodyStates[BodyName::BodyRightFoot].getPos();
				break;
		default:
			break;
		}*/

		switch (i)
		{
			case BodyName::BodyLeftArm:
				i = BodyName::BodyLeftHand;
				break;
			case BodyName::BodyLeftLeg:
				i = BodyName::BodyLeftFoot;
				break;
			case BodyName::BodyRightArm:
				i = BodyName::BodyRightHand;
				break;
			case BodyName::BodyRightLeg:
				i = BodyName::BodyRightFoot;
				break;
		default:
			break;
		}

		if (i == BodyName::BodyTrunk)
			i = BodyName::BodySpine;
		if (i == BodyName::BodyTrunkUpper)
			i = BodyName::BodyTrunk;
		
		Vector3 targetDirVector = getBodyDirectionZ(i);

		if (i == BodyName::BodyTrunkLower)
			i = BodyName::BodySpine;

		Vector3 mPos = bodyStates[i].getPos();

		float bone_size = bodyStates[i].getBoneSize() / 2;
		Vector3 ePos_i(mPos[0] + targetDirVector.x() * bone_size, mPos[1] + targetDirVector.y() * bone_size, mPos[2] + targetDirVector.z() * bone_size);

		return ePos_i;
	}
};

class mFileHandler
{
private:
	std::string mfilename;
	bool mExists;
	FILE * rwFileStream;

	bool mOpenCreateFile(const std::string& name) 
	{
		fopen_s(&rwFileStream, name.c_str(), "r");
		if (rwFileStream == nullptr) 
		{
			fopen_s(&rwFileStream, name.c_str(), "w+");
		}
		return true;
	}

public:

	bool reWriteFile(std::vector<float> iValues)
	{
		if (mExists)
		{
			fclose(rwFileStream);
			mExists = false;
		}
		fopen_s(&rwFileStream , mfilename.c_str(), "w");
		for (unsigned int i = 0; i < iValues.size(); i++)
		{
			fprintf(rwFileStream, "%f \n", iValues[i]);
		}
		mExists = true;
		return true;
	}

	std::vector<float> readFile()
	{
		std::vector<float> values;
		if (!mExists)
		{
			return values;
		}
		while (!feof(rwFileStream))
		{
			char buff[100];
			char* mLine = fgets(buff, 100, rwFileStream);
			if (mLine)
			{
				values.push_back(stof(buff));
			}
		}
		return values;
	}

	mFileHandler(const std::string& iFilename)
	{
		mfilename = iFilename;
		mExists = mOpenCreateFile(iFilename);
	}

	~mFileHandler()
	{
		if (mExists)
		{
			fclose(rwFileStream);
		}
	}
};

class SimulationContext
{
public:
	enum JointType{ FixedJoint = 0, OneDegreeJoint = 1, TwoDegreeJoint = 2, ThreeDegreeJoint = 3, BallJoint = 4};
	enum BodyType{BodyCapsule = 0, BodyBox = 1, BodySphere = 2};

	enum ContactPoints{ LeftLeg = 5, RightLeg = 6, LeftArm = 7, RightArm = 8 };
	enum BodyName{ BodyTrunk = 0, BodyLeftThigh = 1, BodyRightThigh = 2, BodyLeftShoulder = 3, BodyRightShoulder = 4
				 , BodyLeftLeg = 5, BodyRightLeg = 6, BodyLeftArm = 7, BodyRightArm = 8, BodyHead = 9, BodySpine = 10
				 , BodyLeftHand = 11, BodyRightHand = 12, BodyLeftFoot = 13, BodyRightFoot = 14
				 , BodyTrunkUpper = 15, BodyTrunkLower = 16, BodyHold = 17 };

	Vector3 initialPosition[contextNUM - 1];
	Vector3 resultPosition[contextNUM - 1];
	std::vector<Vector3> holds_body;

	mFileHandler mDesiredAngleFile;

	bool isTestClimber;

	~SimulationContext()
	{
		mDesiredAngleFile.reWriteFile(desiredAnglesBones);
	}

	SimulationContext(bool testClimber, mEnumTestCaseClimber TestID, mDemoTestClimber DemoID, int _nRows, int _nCols)
		:mDesiredAngleFile("mDesiredAngleFile.txt")
	{
		isTestClimber = testClimber;

		bodyIDs = std::vector<int>(BodyNUM);
		mGeomID = std::vector<int>(BodyNUM); // for drawing stuff

		bodyTypes = std::vector<int>(BodyNUM);
		boneSize = std::vector<float>(BodyNUM);
		fatherBodyIDs = std::vector<int>(BodyNUM);
		jointIDs = std::vector<int>(BodyNUM - 1);
		jointTypes = std::vector<int>(BodyNUM - 1);

		jointHoldBallIDs = std::vector<int>(4);
		holdPosIndex = std::vector<int>(4);

		for (int i = 0; i < 4; i++)
		{
			jointHoldBallIDs[i] = -1;
			holdPosIndex[i] = -1;
		}

		for (int i = 0; i < BodyNUM -1; i++)
			jointIDs[i] = -1;

		maxNumContexts = contextNUM; 
		
		currentFreeSavingStateSlot = 0;

		//Allocate one simulation context for each sample, plus one additional "master" context
		initOde(maxNumContexts);  // contactgroup = dJointGroupCreate (1000000); //might be needed, right now it is 0
		setCurrentOdeContext(ALLTHREADS);
		odeRandSetSeed(0);
		odeSetContactSoftCFM(contactCFM);
		
		odeWorldSetCFM(worldCfM);
		odeWorldSetERP(worldERP);

		odeWorldSetGravity(0, 0, -9.81f);

		int spaceID = odeCreatePlane(0,0,0,1,0);

		if (!testClimber)
		{
			createHumanoidBody(0, -0.5f, 2.3 * boneLength, 70.0f);
		}
		else
		{
			switch (TestID)
			{
			case mEnumTestCaseClimber::TestAngle:
				createHumanoidBody(0, -0.5f, 5 * boneLength, 70.0f);
				break;
			case mEnumTestCaseClimber::TestCntroller:
				createHumanoidBody(0, -0.5f, 2.3 * boneLength, 70.0f); // 
				break;
			}
		}

		// calculate joint size and read desired angles
		mJointSize = 0;
		for (int i = 0; i < BodyNUM - 1; i++)
		{
			mJointSize += jointTypes[i];
			for (int j = 0; j < jointTypes[i]; j++)
			{
				desiredAnglesBones.push_back(0.0f);
			}
		}
		std::vector<float> readFiledAngles = mDesiredAngleFile.readFile();
		if (readFiledAngles.size() > 0)
		{
			for (unsigned int i = 0; i < readFiledAngles.size(); i++)
			{
				desiredAnglesBones[i] = readFiledAngles[i];
			}
		}

		if (!testClimber)
		{
			attachContactPointToHold(ContactPoints::LeftArm, 2);
			attachContactPointToHold(ContactPoints::RightArm, 3);
			attachContactPointToHold(ContactPoints::LeftLeg, 0);
			attachContactPointToHold(ContactPoints::RightLeg, 1);

			createEnvironment(DemoID, _nRows, _nCols);
		}
		else
		{
			Vector3 hPos;
			switch (TestID)
			{
			case mEnumTestCaseClimber::TestAngle:
				hPos = getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
				createJointType(hPos.x(), hPos.y(), hPos.z(), -1, SimulationContext::BodyName::BodyTrunk, JointType::FixedJoint);
				break;
			case mEnumTestCaseClimber::TestCntroller:
//				hPos = getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
//				createJointType(hPos.x(), hPos.y(), hPos.z(), -1, SimulationContext::BodyName::BodyTrunk, JointType::FixedJoint);

				attachContactPointToHold(ContactPoints::LeftArm, 2);
				attachContactPointToHold(ContactPoints::RightArm, 3);
				attachContactPointToHold(ContactPoints::LeftLeg, 0);
				attachContactPointToHold(ContactPoints::RightLeg, 1);

				createEnvironment(DemoID, _nRows, _nCols);
				break;
			}
		}

		for (int i = 0; i < maxNumContexts; i++)
		{
			int cContextSavingSlotNum = getNextFreeSavingSlot();
			saveOdeState(cContextSavingSlotNum);
		}

		//We're done, now we should have nSamples+1 copies of a model
		masterContext = contextNUM - 1;
		setCurrentOdeContext(masterContext);
//		stepOde(timeStep,false);
	}

	static float getAbsAngleBtwVectors(Vector3 v0, Vector3 v1)
	{
		v0.normalize();
		v1.normalize();

		float angle = acosf(v0.x() * v1.x() + v0.y() * v1.y() + v0.z() * v1.z());

		return angle;
	}

	static float getAngleBtwVectorsXZ(Vector3 _v0, Vector3 _v1)
	{
		Vector2 v0(_v0.x(), _v0.z());
		Vector2 v1(_v1.x(), _v1.z());

		v0.normalize();
		v1.normalize();

		float angle = acosf(v0.x() * v1.x() + v0.y() * v1.y());

		float crossproduct = (v0.x() * v1.y() - v0.y() * v1.x()) * 1;// in direction of k

		if (crossproduct < 0)
			angle = -angle;

		return angle;
	}

	void detachContactPoint(ContactPoints iEndPos)
	{
		int cContextNum = getCurrentOdeContext();

		setCurrentOdeContext(ALLTHREADS);

		int jointIndex = iEndPos - ContactPoints::LeftLeg;

		odeJointAttach(jointHoldBallIDs[jointIndex], 0, 0);
//		odeJointAttach(jointHoldIDs[jointIndex], 0, 0);

		holdPosIndex[jointIndex] = -1;

		setCurrentOdeContext(cContextNum);
	}

	void attachContactPointToHold(ContactPoints iEndPos, int iHoldID)
	{
		int cContextNum = getCurrentOdeContext();

		setCurrentOdeContext(ALLTHREADS);

		int jointIndex = iEndPos - ContactPoints::LeftLeg;
		int boneIndex = BodyName::BodyLeftLeg + jointIndex;
		Vector3 hPos = getEndPointPosBones(boneIndex);

		switch (boneIndex)
		{
			case BodyName::BodyLeftArm:
				boneIndex = BodyName::BodyLeftHand;
				break;
			case BodyName::BodyLeftLeg:
				boneIndex = BodyName::BodyLeftFoot;
				break;
			case BodyName::BodyRightArm:
				boneIndex = BodyName::BodyRightHand;
				break;
			case BodyName::BodyRightLeg:
				boneIndex = BodyName::BodyRightFoot;
				break;
		default:
			break;
		}

		if (jointHoldBallIDs[jointIndex] == -1) // create the hold joint only once
		{
			int cJointBallID = createJointType(hPos.x(), hPos.y(), hPos.z(), -1, boneIndex);

			jointHoldBallIDs[jointIndex] = cJointBallID;
		}
		else
		{
			odeJointAttach(jointHoldBallIDs[jointIndex], 0, bodyIDs[boneIndex]);
			odeJointSetBallAnchor(jointHoldBallIDs[jointIndex], hPos.x(), hPos.y(), hPos.z());
		}

		holdPosIndex[jointIndex] = iHoldID;

		setCurrentOdeContext(cContextNum);
	}

	int getNextFreeSavingSlot() 
	{
		return currentFreeSavingStateSlot++;
	}

	int getMasterContextID()
	{
		return masterContext;
	}

	////////////////////////////////////////// get humanoid and hold bodies info /////////////////////////

	bool checkViolatingRelativeDis()
	{
		for (unsigned int i = 0; i < fatherBodyIDs.size(); i++)
		{
			if (fatherBodyIDs[i] != -1)
			{
				Vector3 bone_i = getEndPointPosBones(i, true);
				Vector3 f_bone_i = getEndPointPosBones(fatherBodyIDs[i], true);

				float coeff = 1.5f;

				if (i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightShoulder)
					coeff = 2.0f;

				if ((bone_i - f_bone_i).norm() > (coeff * boneSize[i]))
				{
					return true;
				}
			}
		}

		return false;
	}

	Vector3 getBodyDirectionZ(int i)
	{
		dMatrix3 R;
		ConstOdeQuaternion mQ;
		if (i == BodyName::BodyTrunkLower)
			mQ = odeBodyGetQuaternion(bodyIDs[BodyName::BodySpine]);
		else
			mQ = odeBodyGetQuaternion(bodyIDs[i]);
		dRfromQ(R, mQ);

		int targetDirection = 2; // alwayse in z direction
		int t_i = i;
		if (i == BodyName::BodyTrunkLower)
			t_i = BodyName::BodySpine;
		if (bodyTypes[t_i] == BodyType::BodyBox)
		{
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder)
				targetDirection = 0; // unless one of these bones is wanted
		}
		Vector3 targetDirVector(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
		if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyLeftThigh || i == BodyName::BodyRightThigh 
			|| i == BodyName::BodyLeftLeg || i == BodyName::BodyRightLeg || i == BodyName::BodyTrunkLower || i == BodyName::BodyLeftHand
			|| i == BodyName::BodyLeftFoot || i == BodyName::BodyRightFoot)
		{
			targetDirVector = -targetDirVector;
		}

		return targetDirVector;
	}

	int getHoldBodyIDs(int i)
	{
		return holdPosIndex[i];
	}

	int getHoldBodyIDsSize()
	{
		return holdPosIndex.size();
	}

	int getJointSize()
	{
		return mJointSize;
	}

	Vector3 getBonePosition(int i)
	{
		ConstOdeVector rPos;
		
		if (i < BodyNUM)
			rPos = odeBodyGetPosition(bodyIDs[i]);
		else
		{
			//rPos = odeBodyGetPosition(bodyHoldIDs[i - BodyNUM]);
			return Vector3(0.0f,0.0f,0.0f);
		}
		
		return Vector3(rPos[0], rPos[1], rPos[2]);
	}
	
	Vector3 getBoneLinearVelocity(int i)
	{
		ConstOdeVector rVel = odeBodyGetLinearVel(bodyIDs[i]);
		
		return Vector3(rVel[0], rVel[1], rVel[2]);
	}

	Vector4 getBoneAngle(int i)
	{
		ConstOdeQuaternion rAngle = odeBodyGetQuaternion(bodyIDs[i]);

		return Vector4(rAngle[0], rAngle[1], rAngle[2], rAngle[3]);
	}

	Vector3 getBoneAngularVelocity(int i)
	{
		ConstOdeVector rAVel = odeBodyGetAngularVel(bodyIDs[i]);
		
		return Vector3(rAVel[0], rAVel[1], rAVel[2]);
	}

	float getJointAngle(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float angle = odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			return angle;
		}
		return odeJointGetHingeAngle(jointIDs[jointIDIndex[i]]);
	}
	float getJointAngleRate(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float angle = odeJointGetAMotorAngleRate(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			return angle;
		}
		return odeJointGetHingeAngleRate(jointIDs[jointIDIndex[i]]);
	}
	float getJointFMax(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			return odeJointGetAMotorParam(jointIDs[jointIDIndex[i]], dParamFMax1 + dParamGroup*jointAxisIndex[i]);
		}
		return odeJointGetHingeParam(jointIDs[jointIDIndex[i]], dParamFMax1 );
	}


	float getJointAngleMin(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			return odeJointGetAMotorParam(jointIDs[jointIDIndex[i]], dParamLoStop1 + dParamGroup*jointAxisIndex[i]);
		}
		return odeJointGetHingeParam(jointIDs[jointIDIndex[i]], dParamLoStop1 );
	}
	float getJointAngleMax(int i)
	{
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			return odeJointGetAMotorParam(jointIDs[jointIDIndex[i]], dParamHiStop1 + dParamGroup*jointAxisIndex[i]);
		}
		return odeJointGetHingeParam(jointIDs[jointIDIndex[i]], dParamHiStop1 );
	}


	Vector3 getEndPointPosBones(int i, bool flag_calculate_exact_val = false)
	{
		ConstOdeVector mPos;
		/*switch (i)
		{
			case BodyName::BodyLeftArm:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyLeftHand]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
			case BodyName::BodyLeftLeg:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyLeftFoot]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
			case BodyName::BodyRightArm:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyRightHand]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
			case BodyName::BodyRightLeg:
				mPos = odeBodyGetPosition(bodyIDs[BodyName::BodyRightFoot]);
				return Vector3(mPos[0], mPos[1], mPos[2]);
				break;
		default:
			break;
		}*/
		if (!flag_calculate_exact_val)
		{
			switch (i)
			{
				case BodyName::BodyLeftArm:
					i = BodyName::BodyLeftHand;
					break;
				case BodyName::BodyLeftLeg:
					i = BodyName::BodyLeftFoot;
					break;
				case BodyName::BodyRightArm:
					i = BodyName::BodyRightHand;
					break;
				case BodyName::BodyRightLeg:
					i = BodyName::BodyRightFoot;
					break;
			default:
				break;
			}
		}
		if (i == BodyName::BodyTrunk)
			i = BodyName::BodySpine;
		if (i == BodyName::BodyTrunkUpper)
			i = BodyName::BodyTrunk;
		
		Vector3 targetDirVector = getBodyDirectionZ(i);

		if (i == BodyName::BodyTrunkLower)
			i = BodyName::BodySpine;

		mPos = odeBodyGetPosition(bodyIDs[i]);

		float bone_size = boneSize[i] / 2;
		Vector3 ePos_i(mPos[0] + targetDirVector.x() * bone_size, mPos[1] + targetDirVector.y() * bone_size, mPos[2] + targetDirVector.z() * bone_size);

		return ePos_i;
	}

	/////////////////////////////////////////// setting motor speed to control humanoid body /////////////

	void setMotorSpeed(int i, float iSpeed)
	{
		if (jointIDs[jointIDIndex[i]] == -1)
			return;
		if ((jointIDIndex[i] + 1) == BodyName::BodyHead || (jointIDIndex[i] + 1) == BodyName::BodyLeftHand || (jointIDIndex[i] + 1) == BodyName::BodyRightHand )
		{
			float angle=odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]],jointAxisIndex[i]);
			iSpeed=-angle; //p control, keep head and wriat zero rotation
			//Vector3 dir_trunk = getBodyDirectionZ(BodyName::BodyTrunk);
			//Vector3 dir_head = getBodyDirectionZ(BodyName::BodyTrunk);
			//Vector3 diff_head_trunk = dir_trunk - dir_head;
			//diff_head_trunk[2] += FLT_EPSILON;
			//switch (jointAxisIndex[i])
			//{
			//case 0:// rotation about local x
			//	diff_head_trunk[0] = 0;
			//	diff_head_trunk.normalize();
			//	iSpeed = atan2(diff_head_trunk[1], diff_head_trunk[2]);
			//	break;
			//case 1:
			//	diff_head_trunk[1] = 0;
			//	diff_head_trunk.normalize();
			//	iSpeed = atan2(diff_head_trunk[0], diff_head_trunk[2]);
			//	break;
			//case 2:
			//	iSpeed = 0;
			//	break;
			//}
		}
		Vector2 mLimits = jointLimits[i];
		const float angleLimitBuffer=deg2rad*2.5f;
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float cAngle = odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			float nAngle = cAngle + (float)(nPhysicsPerStep * iSpeed * timeStep);
			if (((nAngle < mLimits[0]+angleLimitBuffer) && iSpeed<0)
				|| ((nAngle > mLimits[1]+angleLimitBuffer) && iSpeed>0))
			{
				iSpeed = 0;
			}
			switch (jointAxisIndex[i])
			{
			case 0:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel1, iSpeed);
				break;
			case 1:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel2, iSpeed);
				break;
			case 2:
				odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel3, iSpeed);
				break;
			}
			return;
		}
		float cAngle = odeJointGetHingeAngle(jointIDs[jointIDIndex[i]]);
		float nAngle = cAngle + (float)(nPhysicsPerStep * iSpeed * timeStep);
		if (((nAngle < mLimits[0]+angleLimitBuffer) && iSpeed<0)
			|| ((nAngle > mLimits[1]+angleLimitBuffer) && iSpeed>0))
		{
			iSpeed = 0;
		}
		odeJointSetHingeParam(jointIDs[jointIDIndex[i]], dParamVel1, iSpeed);
		return;
	}
	void driveMotorToPose(int i, float targetAngle)
	{
		if (jointIDs[jointIDIndex[i]] == -1)
			return;
		if ((jointIDIndex[i] + 1) == BodyName::BodyHead || (jointIDIndex[i] + 1) == BodyName::BodyLeftHand || (jointIDIndex[i] + 1) == BodyName::BodyRightHand )
		{
			float angle=odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]],jointAxisIndex[i]);
			odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel1, -angle);
			odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel2, -angle);
			odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel3, -angle);
		}
		Vector2 mLimits = jointLimits[i];
		if (jointTypes[jointIDIndex[i]] == JointType::ThreeDegreeJoint)
		{
			float cAngle = odeJointGetAMotorAngle(jointIDs[jointIDIndex[i]], jointAxisIndex[i]);
			odeJointSetAMotorParam(jointIDs[jointIDIndex[i]], dParamVel1+jointAxisIndex[i]*dParamGroup, (targetAngle-cAngle)/motorPoseInterpolationTime);
			return;
		}
		float cAngle = odeJointGetHingeAngle(jointIDs[jointIDIndex[i]]);
		odeJointSetHingeParam(jointIDs[jointIDIndex[i]], dParamVel1, (targetAngle-cAngle)/motorPoseInterpolationTime);
	}
	void setFmax(int joint, float fmax)
	{
		if (odeJointGetType(joint)==dJointTypeAMotor)
		{
			odeJointSetAMotorParam(joint,dParamFMax1,fmax);
			odeJointSetAMotorParam(joint,dParamFMax2,fmax);
			odeJointSetAMotorParam(joint,dParamFMax3,fmax);
		}
		else
		{
			odeJointSetHingeParam(joint,dParamFMax1,fmax);
		}
	}
	void setMotorGroupFmaxes(const float *fmax)
	{
		setFmax(getJointBody(BodySpine),std::max(torsoMinFMax,fmax[fmTorso]));
		//setFmax(getJointBody(BodyHead),fmax[fmTorso]);

		setFmax(getJointBody(BodyLeftShoulder),fmax[fmLeftArm]);
		setFmax(getJointBody(BodyLeftArm),fmax[fmLeftArm]);
		setFmax(getJointBody(BodyLeftHand),fmax[fmLeftArm]);

		setFmax(getJointBody(BodyLeftThigh),fmax[fmLeftLeg]);
		setFmax(getJointBody(BodyLeftLeg),fmax[fmLeftLeg]);
		setFmax(getJointBody(BodyLeftFoot),fmax[fmLeftLeg]);

		setFmax(getJointBody(BodyRightShoulder),fmax[fmRightArm]);
		setFmax(getJointBody(BodyRightArm),fmax[fmRightArm]);
		setFmax(getJointBody(BodyRightHand),fmax[fmRightArm]);

		setFmax(getJointBody(BodyRightThigh),fmax[fmRightLeg]);
		setFmax(getJointBody(BodyRightLeg),fmax[fmRightLeg]);
		setFmax(getJointBody(BodyRightFoot),fmax[fmRightLeg]);
	}

	inline static float odeVectorSquaredNorm(ConstOdeVector v)
	{
		return squared(v[0])+squared(v[1])+squared(v[2]);
	}
	float getMotorAppliedSqTorque(int i)
	{
		if (jointIDs[jointIDIndex[i]] == -1)
			return 0;
		return odeVectorSquaredNorm(odeJointGetAccumulatedTorque(jointIDs[jointIDIndex[i]],0))
			+ odeVectorSquaredNorm(odeJointGetAccumulatedTorque(jointIDs[jointIDIndex[i]],1));
	}
	float getSqForceOnFingers()
	{
		float result=0;
		dVector3 f;
		odeBodyGetAccumulatedForce(bodyIDs[BodyName::BodyLeftHand],-1,f);
		result+=odeVectorSquaredNorm(f);
		odeBodyGetAccumulatedForce(bodyIDs[BodyName::BodyRightHand],-1,f);
		result+=odeVectorSquaredNorm(f);
		return result;
		//The following crashes when holds detached, for some reason
		//float result=0;
		//for (int j=2; j<4; j++)
		//{
		//	int hold=getHoldBodyIDs(j); 
		//	if (hold>=0)
		//	{
		//		int joint=jointHoldBallIDs[hold];
		//		if (odeJointGetBody(joint,0)>=0)
		//			result+=odeVectorSquaredNorm(odeJointGetAccumulatedForce(joint,0));
		//		if (odeJointGetBody(joint,1)>=0)
		//			result+=odeVectorSquaredNorm(odeJointGetAccumulatedForce(joint,1));
		//	}
		//}
		//return result;
	}
	float getDesMotorAngleFromID(int i)
	{
		int bodyID = jointIDIndex[i] + 1;
		int axisID = jointAxisIndex[i];
		return getDesMotorAngle(bodyID, axisID);
	}

	float getDesMotorAngle(int &iBodyName, int &iAxisNum)
	{
		int jointIndex = iBodyName - 1;
		if (jointIndex > (int)(jointIDs.size()-1))
		{
			jointIndex = jointIDs.size()-1;
		}
		if (jointIndex < 0)
		{
			jointIndex = 0;
		}
		iBodyName = jointIndex + 1;

		if (jointTypes[jointIndex] == JointType::ThreeDegreeJoint)
		{
			if (iAxisNum > 2)
			{
				iAxisNum = 0;
			}
			if (iAxisNum < 0)
			{
				iAxisNum = 2;
			}
		}
		else
		{
			if (iAxisNum != 0)
			{
				iAxisNum = 0;
			}
		}

		int m_angle_index = 0;
		for (int b = 0; b < BodyNUM; b++)
		{
			for (int j = 0; j < jointTypes[b]; j++)
			{
				if (b == iBodyName - 1 && j == iAxisNum)
				{
					return desiredAnglesBones[m_angle_index];
				}
				m_angle_index++;
			}
		}
		return 0.0f;
	}

	void setMotorAngle(int &iBodyName, int &iAxisNum, float& dAngle)
	{
		int jointIndex = iBodyName - 1;
		if (jointIndex > (int)(jointIDs.size()-1))
		{
			jointIndex = jointIDs.size()-1;
		}
		if (jointIndex < 0)
		{
			jointIndex = 0;
		}
		iBodyName = jointIndex + 1;

		if (jointTypes[jointIndex] == JointType::ThreeDegreeJoint)
		{
			if (iAxisNum > 2)
			{
				iAxisNum = 0;
			}
			if (iAxisNum < 0)
			{
				iAxisNum = 2;
			}
		}
		else
		{
			if (iAxisNum != 0)
			{
				iAxisNum = 0;
			}
		}

		int m_angle_index = 0;
		for (int b = 0; b < BodyNUM; b++)
		{
			int mJointIndex = b - 1;
			if (mJointIndex < 0)
			{
				continue;
			}
			if (jointIDs[mJointIndex] == -1)
			{
				m_angle_index += jointTypes[mJointIndex];
				continue;
			}
			float source_angle = 0.0f;
			for (int axis = 0; axis < jointTypes[mJointIndex]; axis++)
			{
				if (jointTypes[mJointIndex] == JointType::ThreeDegreeJoint)
				{
					source_angle = odeJointGetAMotorAngle(jointIDs[mJointIndex],axis);
				}
				else
				{
					source_angle = odeJointGetHingeAngle(jointIDs[mJointIndex]);
				}

				if (b == iBodyName && axis == iAxisNum)
				{
					desiredAnglesBones[m_angle_index] = dAngle;
				}

				float iSpeed = desiredAnglesBones[m_angle_index] - source_angle;
				m_angle_index++;

				if (fabs(iSpeed) > maxSpeed)
				{
					iSpeed = fsign(iSpeed) * maxSpeed;
				}

				if (jointTypes[mJointIndex] == JointType::ThreeDegreeJoint)
				{
					switch (axis)
					{
					case 0:
						odeJointSetAMotorParam(jointIDs[mJointIndex], dParamVel1, iSpeed);
						break;
					case 1:
						odeJointSetAMotorParam(jointIDs[mJointIndex], dParamVel2, iSpeed);
						break;
					case 2:
						odeJointSetAMotorParam(jointIDs[mJointIndex], dParamVel3, iSpeed);
						break;
					}
				}
				else
				{
					odeJointSetHingeParam(jointIDs[mJointIndex], dParamVel1, iSpeed);
				}
			}
		}

		return;
	}

	void saveContextIn(BipedState& c)
	{
		for (int i = 0; i < BodyNUM; i++)
		{
			if (c.bodyStates.size() < BodyNUM)
			{
				c.bodyStates.push_back(BodyState());
			}
		}

		for (int i = 0; i < BodyNUM; i++)
		{
			c.bodyStates[i].setPos(getBonePosition(i));
			c.bodyStates[i].setAngle(getBoneAngle(i));
			c.bodyStates[i].setVel(getBoneLinearVelocity(i));
			c.bodyStates[i].setAVel(getBoneAngularVelocity(i));
			c.bodyStates[i].setBoneSize(boneSize[i]);
			c.bodyStates[i].setBodyType(bodyTypes[i]);
			c.bodyStates[i].setBodyMass(odeBodyGetMass(bodyIDs[i]));
		}

		for (int i = 0; i < getJointSize(); i++)
		{
			if ((int)c.toDesAngles.size() < getJointSize())
			{
				c.toDesAngles.push_back(getJointAngle(i));
			}
			else
			{
				c.toDesAngles[i] = getJointAngle(i);
			}
		}

		return;
	}

	//////////////////////////////////////////// for drawing bodies ///////////////////////////////////////

	void mDrawStuff(int iControlledBody, bool whileOptimizing=false, bool drawBody = true)//, double cTimeElapsed, float numMovingLimbs, int pathNum)
	{
		if (!whileOptimizing)
			setCurrentOdeContext(masterContext);

		//dsMyPushMatrix();
		//dsMyRotateZ(90.0f);

		//dsSetTexture (DS_WOOD);
		
		for (int i = 0; i < BodyNUM && drawBody; i++)
		{
			rcSetColor(1,1,1,1.0f);
			if (i == iControlledBody && iControlledBody >= 0 && isTestClimber)
			{
				rcSetColor(0,1,0,1.0f);
			}
			if (bodyTypes[i] == BodyType::BodyBox)
			{
				float lx, ly, lz;
				odeGeomBoxGetLengths(mGeomID[i], lx, ly, lz);
				float sides[] = {lx, ly, lz};
				rcDrawBox(odeBodyGetPosition(bodyIDs[i]), odeBodyGetRotation(bodyIDs[i]), sides);
			}
			else if (bodyTypes[i] == BodyType::BodyCapsule)
			{
				float radius,length;
				odeGeomCapsuleGetParams(mGeomID[i], radius, length);
				rcDrawCapsule(odeBodyGetPosition(bodyIDs[i]), odeBodyGetRotation(bodyIDs[i]), length, radius);
			}
			else
			{
				float radius = odeGeomSphereGetRadius(mGeomID[i]);
				rcDrawSphere(odeBodyGetPosition(bodyIDs[i]), odeBodyGetRotation(bodyIDs[i]), radius);
			}
		}
	
		if (!whileOptimizing)
		{
			//dsSetTexture(DS_TEXTURE_NUMBER::DS_NONE);
			for (unsigned int i = 0; i < mENVGeomTypes.size(); i++)
			{
				if (mENVGeomTypes[i] == BodyType::BodyBox)
				{
					rcSetColor(1,1,1,1.0f);
					float lx, ly, lz;
					odeGeomBoxGetLengths(mENVGeoms[i], lx, ly, lz);
					float sides[] = {lx, ly, lz};
					rcDrawBox(odeGeomGetPosition(mENVGeoms[i]), odeGeomGetRotation(mENVGeoms[i]), sides); 
				}
				else
				{
					rcSetColor(1,1,1,0.5f);
					float radius = odeGeomSphereGetRadius(mENVGeoms[i]) / 2.0f;
					rcDrawSphere(odeGeomGetPosition(mENVGeoms[i]), odeGeomGetRotation(mENVGeoms[i]), radius);
				}
			}

			rcSetColor(1,0,0,1.0f);
			for (unsigned int i = 0; i < BodyNUM && drawBody; i++)
			{
				Vector3 dir_i = getBodyDirectionY(i);

				Vector3 pos_i = getBonePosition(i);

				Vector3 n_pos_i = pos_i + 0.5f * dir_i;
				if (i == BodyName::BodyTrunk)
				{
					float p1[] = {pos_i.x(),pos_i.y(),pos_i.z()};
					float p2[] = {n_pos_i.x(),n_pos_i.y(),n_pos_i.z()};
					rcDrawLine(p1, p2);
				}
			}
		}

	//	rcPrintString("Current Time:%f \n", cTimeElapsed);
	//	rcPrintString("Path %d with %d MovingLimbs \n", pathNum, (int)numMovingLimbs);
		//dsMyPopMatrix();
		return;
	}
	
	Vector3 getGoalPos()
	{
		return goal_pos;
	}

	Vector3 getBodyDirectionY(int i)
	{
		dMatrix3 R;
		ConstOdeQuaternion mQ;
		if (i == BodyName::BodyTrunkLower)
			mQ = odeBodyGetQuaternion(bodyIDs[BodyName::BodySpine]);
		else
			mQ = odeBodyGetQuaternion(bodyIDs[i]);
		dRfromQ(R, mQ);

		int targetDirection = 1; // alwayse in y direction
		int t_i = i;
		if (i == BodyName::BodyTrunkLower)
			t_i = BodyName::BodySpine;
		if (bodyTypes[t_i] == BodyType::BodyBox)
		{
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder)
				targetDirection = 0; // unless one of these bones is wanted
		}
		Vector3 targetDirVector(R[4 * 0 + targetDirection], R[4 * 1 + targetDirection], R[4 * 2 + targetDirection]);
		if (i == BodyName::BodyTrunkLower)
		{
			targetDirVector = -targetDirVector;
		}

		return targetDirVector;
	}

	float getClimberRadius()
	{
		return climberRadius;
	}

	float getClimberLegLegDis()
	{
		return climberLegLegDis;
	}

	float getClimberHandHandDis()
	{
		return climberHandHandDis;
	}


	void createEnvironment(mDemoTestClimber DemoID, int _nRows, int _nCols)
	{
		Vector3 middleWallDegree = createWall(DemoID);

		float betweenHolds = 0.2f;

		Vector3 cPos;
		Vector3 rLegPos;
		Vector3 lLegPos;

		float cHeightZ = 0.35f;
		float cWidthX = -FLT_MAX;

		float wall_1_z = 0.195f;
		float wall_1_x = 0.245f;
		float btwHolds1 = 0.2f;

		float wall_2_z = 1.37f;
		float wall_2_x = 0.25f;
		float btwHolds2_middle = 0.2f;
		float btwHolds2_topRow = 0.22f;

		float wall_3_z = 2.495f;
		float wall_3_x = 0.3f;
		float btwHolds3 = 0.17f;

		int counter = 0;
		
		Vector3 Origin(middleWallDegree.x(), 0.0f, middleWallDegree.y());
		float theta = middleWallDegree.z();
		Vector3 dPos;
		Vector3 nPos;
		float _disBtw = 0.75f;
		switch (DemoID)
		{
		case mDemoTestClimber::DemoRoute1:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 4 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 3 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(cPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += btwHolds2_topRow;
			addHoldBodyToWorld(cPos); // helper 2

			cPos = lLegPos;
			cPos[2] = wall_3_z + btwHolds3;
			cPos[0] += (4 * btwHolds3 - 2 * btwHolds1);
			addHoldBodyToWorld(cPos); // helper 3

			cPos[2] += 3 * btwHolds3;
			cPos[0] += (1 * btwHolds3);
			addHoldBodyToWorld(cPos); // goal

			break;
		case mDemoTestClimber::DemoRoute2:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 3 * btwHolds1;
			addHoldBodyToWorld(cPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z;
			cPos[0] += 9 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // helper 2

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle;
			cPos[0] += (5 * btwHolds2_middle);
			addHoldBodyToWorld(cPos); // helper 3

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += (2 * btwHolds2_middle + btwHolds2_topRow);
			addHoldBodyToWorld(cPos); // helper 4

			cPos = lLegPos;
			cPos[2] = wall_3_z + 2 * btwHolds3;
			cPos[0] += (9 * btwHolds3);
			addHoldBodyToWorld(cPos); // helper 5

			cPos[2] += 2 * btwHolds3;
			addHoldBodyToWorld(cPos); // goal

			break;
		case mDemoTestClimber::DemoRoute3:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 3 * btwHolds1;

			addHoldBodyToWorld(cPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle;
			cPos[0] += (5 * btwHolds2_middle);

			addHoldBodyToWorld(cPos); // helper 3

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += (2 * btwHolds2_middle + btwHolds2_topRow);

			addHoldBodyToWorld(cPos); // helper 4

			cPos = lLegPos;
			cPos[2] = wall_3_z + 2 * btwHolds3;
			cPos[0] += (9 * btwHolds3);

			addHoldBodyToWorld(cPos); // helper 5

			cPos[2] += 2 * btwHolds3;

			addHoldBodyToWorld(cPos); // goal
			break;
		case mDemoTestClimber::Demo45Wall2:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = wall_1_z;
			addHoldBodyToWorld(cPos); // leftLeg
			lLegPos = cPos;

			rLegPos = cPos;
			rLegPos[0] += 2 * btwHolds1;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] = wall_2_z - 0.2 * sinf(theta);
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * btwHolds2_middle;
			addHoldBodyToWorld(cPos); // RightHand

			cPos = lLegPos;
			cPos[2] += 3 * btwHolds1;
			cPos[0] += 3 * btwHolds1;
			if (cPos[2] > 2.0f)
			{
				dPos = cPos - Origin;
				nPos = Origin + Vector3(dPos.x()
									   ,dPos.y() * cosf(theta) - dPos.z() * sinf(theta)
									   ,dPos.y() * sinf(theta) + dPos.z() * cosf(theta));
			}
			else
			{
				nPos = cPos;
			}
			addHoldBodyToWorld(nPos); // helper 1

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle;
			cPos[0] += (5 * btwHolds2_middle);

			if (cPos[2] > 2.0f)
			{
				dPos = cPos - Origin;
				nPos = Origin + Vector3(dPos.x()
									   ,dPos.y() * cosf(theta) - dPos.z() * sinf(theta)
									   ,dPos.y() * sinf(theta) + dPos.z() * cosf(theta));
			}
			else
			{
				nPos = cPos;
			}

			addHoldBodyToWorld(nPos); // helper 3

			cPos = lLegPos;
			cPos[2] = wall_2_z + 3 * btwHolds2_middle + btwHolds2_topRow;
			cPos[0] += (2 * btwHolds2_middle + btwHolds2_topRow);

			if (cPos[2] >= 2.0f)
			{
				dPos = cPos - Origin;
				nPos = Origin + Vector3(dPos.x()
									   ,dPos.y() * cosf(theta) - dPos.z() * sinf(theta)
									   ,dPos.y() * sinf(theta) + dPos.z() * cosf(theta));
			}
			else
			{
				nPos = cPos;
			}

			addHoldBodyToWorld(nPos); // helper 4

			cPos = cPos + Vector3(-_disBtw * cosf(theta), 0.0f, _disBtw * sinf(theta));

			dPos = cPos - Origin;
			nPos = Origin + Vector3(dPos.x()
								   ,dPos.y() * cosf(theta) - dPos.z() * sinf(theta)
								   ,dPos.y() * sinf(theta) + dPos.z() * cosf(theta));

			addHoldBodyToWorld(nPos); // helper 4

			cPos = cPos + Vector3(-_disBtw * cosf(theta), 0.0f, _disBtw * sinf(theta));

			dPos = cPos - Origin;
			nPos = Origin + Vector3(dPos.x()
								   ,dPos.y() * cosf(theta) - dPos.z() * sinf(theta)
								   ,dPos.y() * sinf(theta) + dPos.z() * cosf(theta));

			addHoldBodyToWorld(nPos); // helper 5

			cPos = cPos + Vector3(-_disBtw * cosf(theta), 0.0f, _disBtw * sinf(theta));

			dPos = cPos - Origin;
			nPos = Origin + Vector3(dPos.x()
								   ,dPos.y() * cosf(theta) - dPos.z() * sinf(theta)
								   ,dPos.y() * sinf(theta) + dPos.z() * cosf(theta));

			addHoldBodyToWorld(nPos); // helper 6

			cPos = cPos + Vector3(-_disBtw * cosf(theta), 0.0f, _disBtw * sinf(theta));

			dPos = cPos - Origin;
			nPos = Origin + Vector3(dPos.x()
								   ,dPos.y() * cosf(theta) - dPos.z() * sinf(theta)
								   ,dPos.y() * sinf(theta) + dPos.z() * cosf(theta));

			addHoldBodyToWorld(nPos); // goal

			cPos = nPos;
			break;
		case mDemoTestClimber::DemoLongWall:
			rLegPos = getEndPointPosBones(SimulationContext::BodyName::BodyRightLeg);
			lLegPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			
			while (cHeightZ < 10.0f)
			{
				float cDisX = 0.0f;
				float _minZ = FLT_MAX;
				while (cDisX < 1.0f)
				{
					float r1 = (float)rand() / (float)RAND_MAX;
					float r2 = (float)rand() / (float)RAND_MAX;
					cPos = Vector3(lLegPos.x() - 0.4f + r1 * 0.2f + cDisX, 0.0f, cHeightZ + 0.3f * r2);
					addHoldBodyToWorld(cPos);
					cDisX += 0.8f;
					_minZ = min<float>(_minZ, cHeightZ + 0.3f * r2);
					cWidthX = max<float>(cWidthX, cPos.x());
				}
				cHeightZ = _minZ + climberRadius / 2.5f;
			}
			cHeightZ = 0.35f;
			while (cHeightZ < 10.0f)
			{
				float cDisX = 0.0f;
				float _minZ = FLT_MAX;
				while (cDisX < 1.0f)
				{
					float r1 = (float)rand() / (float)RAND_MAX;
					float r2 = (float)rand() / (float)RAND_MAX;
					cPos = Vector3(cWidthX + climberLegLegDis - 0.4f + r1 * 0.2f + cDisX, 0.0f, cHeightZ + 0.3f * r2);
					addHoldBodyToWorld(cPos);
					cDisX += 0.8f;
					_minZ = min<float>(_minZ, cHeightZ + 0.3f * r2);
				}
				cHeightZ = _minZ + climberRadius / 2.5f;
			}

			break;
		case mDemoTestClimber::Demo45Wall:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = 0.35f;
			addHoldBodyToWorld(cPos); // leftLeg

			rLegPos = cPos;
			rLegPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(rLegPos); // rightLeg

			cPos[2] += 3 * betweenHolds + 0.36f;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand

			cPos[2] = middleWallDegree.y() + betweenHolds * sinf(middleWallDegree.z());
			cPos[1] += -betweenHolds * cosf(middleWallDegree.z());
			addHoldBodyToWorld(cPos); // RightHand 1

			cPos[0] -= 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // LeftHand 1

			cPos[2] += 4 * betweenHolds * sinf(middleWallDegree.z());
			cPos[1] += -4 * betweenHolds * cosf(middleWallDegree.z());
			addHoldBodyToWorld(cPos); // LeftHand 2

			cPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand 2

			cPos[2] += 4 * betweenHolds * sinf(middleWallDegree.z());
			cPos[1] += -4 * betweenHolds * cosf(middleWallDegree.z());
			addHoldBodyToWorld(cPos); // LeftHand 2

			cPos[0] -= 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand 2

			cPos[2] += 4 * betweenHolds * sinf(middleWallDegree.z());
			cPos[1] += -4 * betweenHolds * cosf(middleWallDegree.z());
			addHoldBodyToWorld(cPos); // LeftHand 2

			cPos[0] += 2 * betweenHolds;
			addHoldBodyToWorld(cPos); // RightHand 2
			break;
		case mDemoTestClimber::Line:
			cPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);
			cPos[1] = 0.0f;
			cPos[2] = 0.35f;
			addHoldBodyToWorld(cPos); // leftLeg

			cPos[2] += 0.5f;
			addHoldBodyToWorld(cPos); // rightLeg

			cPos[2] += 0.5f;
			addHoldBodyToWorld(cPos); // LeftHand

			cPos[2] += 0.5f;
			addHoldBodyToWorld(cPos); // rightHand

			cPos[2] += 0.5f;
			addHoldBodyToWorld(cPos); // 

			cPos[2] += 0.5f;
			addHoldBodyToWorld(cPos); // 

			cPos[2] += 0.5f;
			addHoldBodyToWorld(cPos); //

			cPos[2] += 0.5f;
			addHoldBodyToWorld(cPos); // 
			break;
		case mDemoTestClimber::DemoLongWallTest:
			rLegPos = getEndPointPosBones(SimulationContext::BodyName::BodyRightLeg);
			lLegPos = getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg);

			while (counter < _nRows)
			{
				int counter_col = 0;
				float cDisX = 0.0f;
				float _minZ = FLT_MAX;
				
				while (counter_col < _nCols)
				{
					float r1 = (float)rand() / (float)RAND_MAX;
					float r2 = (float)rand() / (float)RAND_MAX;
					if (counter_col < 2)
					{
						cPos = Vector3(lLegPos.x() - 0.4f + r1 * 0.2f + cDisX, 0.0f, cHeightZ + 0.3f * r2);
					}
					else
					{
						cPos = Vector3(cWidthX + (climberLegLegDis) - 0.4f + r1 * 0.2f + cDisX, 0.0f, cHeightZ + 0.3f * r2);
					}
					addHoldBodyToWorld(cPos);

					counter_col++;
					
					cDisX += 0.8f;
					
					_minZ = min<float>(_minZ, cHeightZ + 0.3f * r2);
					if (counter_col < 3)
					{
						cWidthX = max<float>(cWidthX, cPos.x());
					}
					if (counter_col == 2)
					{
						cDisX = 0;
					}
				}

				counter++;
				if (_nCols == 1)
				{
					cHeightZ = _minZ + climberRadius / 3.0f;
				}
				else
				{
					cHeightZ = _minZ + climberRadius / 2.5f;
				}
			}
			break;
		default:
			break;
		};

		goal_pos = cPos;
	}

	Vector3 createWall(mDemoTestClimber DemoID)
	{
		if (!(DemoID == mDemoTestClimber::Demo45Wall || DemoID == mDemoTestClimber::Demo45Wall2))
		{
			int cGeomID = odeCreateBox(10, 2 * boneRadius, DemoID == mDemoTestClimber::DemoLongWallTest ? 300 : 15);
			odeGeomSetPosition(cGeomID, 0, (1.5) * boneRadius + mBiasWallY + 0.5f * boneRadius, 5.0f);
			
			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyBox);

			return Vector3(0.0f,0.0f,0.0f);
		}
		else
		{
			// starting from vertical wall
			int cGeomID = odeCreateBox(5, 2 * boneRadius, 2);
			odeGeomSetPosition(cGeomID, 0, (1.5) * boneRadius + mBiasWallY + 0.5f * boneRadius, 1.0f);
			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyBox);

			// going to 45 degree wall
			float l = 5.0f;
			float angle = PI/4;
			float x_b = ((1.5) * boneRadius + mBiasWallY + 0.5f * boneRadius); // -boneRadius -0.5f * l * sinf(angle)
			cGeomID = odeCreateBox(5, 2 * boneRadius, l);
			odeGeomSetPosition(cGeomID, 0,  x_b - 0.5f * (l) * sinf(angle), 2 + 0.5f * boneRadius + 0.5f * (l) * cosf(angle)); // - tanf(angle) * x_b
			
			dMatrix3 R;
			dRFromAxisAndAngle(R, 1, 0, 0, angle);
			odeGeomSetRotation(cGeomID, R);

			mENVGeoms.push_back(cGeomID);
			mENVGeomTypes.push_back(BodyType::BodyBox);

			return Vector3(0.0f, 2.0f, PI/2 - angle);
		}

		return Vector3(0.0f,0.0f,0.0f);
	}

	void addHoldBodyToWorld(Vector3 cPos)
	{
		float iX = cPos.x(), iY = cPos.y(), iZ = cPos.z();

//		std::vector<int> envBodyIndex = createBodyi(-1, holdSize, holdSize, BodyType::BodySphere);
		int cGeomID = odeCreateSphere(holdSize / 2);

		odeGeomSetPosition(cGeomID, iX, iY, iZ);
//		mENVBodies.push_back(envBodyIndex[0]);
		mENVGeoms.push_back(cGeomID);
		mENVGeomTypes.push_back(BodyType::BodySphere);

		odeGeomSetCollideBits (cGeomID, 0); // do not collide with anything!
		odeGeomSetCategoryBits (cGeomID, 0); // do not collide with anything!

//		createJointType(iX, iY, iZ, -1, BodyNUM + mENVBodies.size()-1, JointType::FixedJoint);

		holds_body.push_back(Vector3(iX, iY, iZ));
	}

	int createJointType(float pX, float pY, float pZ, int pBodyNum, int cBodyNum, JointType iJ = JointType::BallJoint)
	{
		//float positionSpring = 10000.0f, stopSpring = 20000.0f, damper = 1.0f, maximumForce = 200.0f;
		float positionSpring = 5000.0f, stopSpring = 5000.0f, damper = 1.0f;
		
		float kp = positionSpring; 
        float kd = damper; 

        float erp = timeStep * kp / (timeStep * kp + kd);
        float cfm = 1.0f / (timeStep * kp + kd);

        float stopDamper = 1.0f; //stops best when critically damped
        float stopErp = timeStep * stopSpring / (timeStep * kp + stopDamper);
        float stopCfm = 1.0f / (timeStep * stopSpring + stopDamper);

		int jointID = -1;
		switch (iJ)
		{
			case JointType::BallJoint:
				jointID = odeJointCreateBall();
			break;
			case JointType::ThreeDegreeJoint:
				jointID = odeJointCreateAMotor();
			break;
			case JointType::TwoDegreeJoint:
				jointID = odeJointCreateHinge2();
			break;
			case JointType::OneDegreeJoint:
				jointID = odeJointCreateHinge();
			break;
			case JointType::FixedJoint:
				jointID = odeJointCreateFixed();
			break;
			default:
				break;
		}
		
		if (pBodyNum >= 0)
		{
			if (cBodyNum < BodyNUM && cBodyNum >= 0)
			{
				odeJointAttach(jointID, bodyIDs[pBodyNum], bodyIDs[cBodyNum]);
			}

			if (iJ == JointType::FixedJoint)
				odeJointSetFixed(jointID);
		}
		else
		{
			if (cBodyNum < BodyNUM && cBodyNum >= 0)
			{
				odeJointAttach(jointID, 0, bodyIDs[cBodyNum]);
			}

			if (iJ == JointType::FixedJoint)
				odeJointSetFixed(jointID);
		}

		float angle = 0;
		switch (iJ)
		{
			case JointType::BallJoint:
				odeJointSetBallAnchor(jointID, pX, pY, pZ);
			break;
			case JointType::ThreeDegreeJoint:
				odeJointSetAMotorNumAxes(jointID, 3);

				if (pBodyNum >= 0)
				{
					odeJointSetAMotorAxis(jointID, 0, 1, 0, 1, 0);
					odeJointSetAMotorAxis(jointID, 2, 2, 1, 0, 0);

					if (cBodyNum == BodyName::BodyLeftShoulder || cBodyNum == BodyName::BodyRightShoulder)
					{
						odeJointSetAMotorAxis(jointID, 0, 1, 0, 1, 0);
						odeJointSetAMotorAxis(jointID, 2, 2, 0, 0, 1);
					}
				}
				else
				{

					odeJointSetAMotorAxis(jointID, 0, 0, 1, 0, 0);
					odeJointSetAMotorAxis(jointID, 2, 2, 0, 0, 1);
				}

				odeJointSetAMotorMode(jointID, dAMotorEuler);

				odeJointSetAMotorParam(jointID, dParamFMax1, maximumForce);
				odeJointSetAMotorParam(jointID, dParamFMax2, maximumForce);
				odeJointSetAMotorParam(jointID, dParamFMax3, maximumForce);

				odeJointSetAMotorParam(jointID, dParamVel1, 0);
				odeJointSetAMotorParam(jointID, dParamVel2, 0);
				odeJointSetAMotorParam(jointID, dParamVel3, 0);

				odeJointSetAMotorParam(jointID, dParamCFM1, cfm);
				odeJointSetAMotorParam(jointID, dParamCFM2, cfm);
				odeJointSetAMotorParam(jointID, dParamCFM3, cfm);

				odeJointSetAMotorParam(jointID, dParamERP1, erp);
				odeJointSetAMotorParam(jointID, dParamERP2, erp);
				odeJointSetAMotorParam(jointID, dParamERP3, erp);

				odeJointSetAMotorParam(jointID, dParamStopCFM1, stopCfm);
				odeJointSetAMotorParam(jointID, dParamStopCFM2, stopCfm);
				odeJointSetAMotorParam(jointID, dParamStopCFM3, stopCfm);

				odeJointSetAMotorParam(jointID, dParamStopERP1, stopErp);
				odeJointSetAMotorParam(jointID, dParamStopERP2, stopErp);
				odeJointSetAMotorParam(jointID, dParamStopERP3, stopErp);

				odeJointSetAMotorParam(jointID, dParamFudgeFactor1, -1);
				odeJointSetAMotorParam(jointID, dParamFudgeFactor2, -1);
				odeJointSetAMotorParam(jointID, dParamFudgeFactor3, -1);
				
			break;
			case JointType::TwoDegreeJoint:
				odeJointSetHinge2Anchor(jointID, pX, pY, pZ);
				odeJointSetHinge2Axis(jointID, 1, 0, 1, 0);
				odeJointSetHinge2Axis(jointID, 2, 0, 0, 1);

				odeJointSetHinge2Param(jointID, dParamFMax1, maximumForce);
				odeJointSetHinge2Param(jointID, dParamFMax2, maximumForce);

				odeJointSetHinge2Param(jointID, dParamVel1, 0);
				odeJointSetHinge2Param(jointID, dParamVel2, 0);

				odeJointSetHinge2Param(jointID, dParamCFM1, cfm);
				odeJointSetHinge2Param(jointID, dParamCFM2, cfm);

				odeJointSetHinge2Param(jointID, dParamStopERP1, stopErp);
				odeJointSetHinge2Param(jointID, dParamStopERP2, stopErp);
				odeJointSetHinge2Param(jointID, dParamFudgeFactor1, -1);
				odeJointSetHinge2Param(jointID, dParamFudgeFactor2, -1);
			break;
			case JointType::OneDegreeJoint:
				odeJointSetHingeAnchor(jointID, pX, pY, pZ);
				if (cBodyNum == BodyName::BodyLeftLeg || cBodyNum == BodyName::BodyRightLeg || cBodyNum == BodyName::BodyRightFoot || cBodyNum == BodyName::BodyLeftFoot)
				{
					odeJointSetHingeAxis(jointID, 1, 0, 0);
				}
				else
				{
					odeJointSetHingeAxis(jointID, 0, 0, 1);
				}
				
				odeJointSetHingeParam(jointID, dParamFMax, maximumForce);

				odeJointSetHingeParam(jointID, dParamVel, 0);

				odeJointSetHingeParam(jointID, dParamCFM, cfm);

				odeJointSetHingeParam(jointID, dParamERP, erp);

				odeJointSetHingeParam(jointID, dParamStopCFM, stopCfm);

				odeJointSetHingeParam(jointID, dParamStopERP, stopErp);

				odeJointSetHingeParam(jointID, dParamFudgeFactor, -1);
			break;
			default:
				break;
		}

		return jointID;
	}

	int getJointBody(BodyName iBodyName)
	{
		return jointIDs[iBodyName - 1];
	}

	void createHumanoidBody(float pX, float pY, float pZ, float dmass)
	{
		float ArmLegWidth = 0.75f * boneRadius;

		float trunkPosX = pX;
		float trunkPosY = pY;
		float trunkPosZ = pZ;
		int cJointID = -1;

		// trunk (root body without parent)
		float trunk_length = 0.1387f + 0.1363f;

		createBodyi(BodyName::BodyTrunk, 1.2f * boneRadius, trunk_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyTrunk], pX, pY, pZ + trunk_length / 2);
		fatherBodyIDs[BodyName::BodyTrunk] = -1;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyTrunk)], unsigned long(0x0080));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyTrunk)], unsigned long(0x36FE));

		// spine
		float spine_length = 0.1625f + 0.091f;

		pZ -= (spine_length / 2);
		createBodyi(BodyName::BodySpine, boneRadius, spine_length);
		odeBodySetPosition(bodyIDs[BodyName::BodySpine], pX, pY, pZ);
		createJointType(pX, pY, pZ + (spine_length / 2), BodyName::BodyTrunk, BodyName::BodySpine);
		cJointID = createJointType(pX, pY, pZ + (spine_length / 2), BodyName::BodyTrunk, BodyName::BodySpine, JointType::ThreeDegreeJoint);
		std::vector<Vector2> cJointLimits = setAngleLimitations(cJointID, BodyName::BodySpine);
		setJointID(BodyName::BodySpine, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodySpine] = BodyName::BodyTrunk;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodySpine)], unsigned long(0x0001));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodySpine)], unsigned long(0x7F6D));

		// left thigh
		float thigh_length = 0.4173f;
		float dis_spine_leg_x = 0.09f;
		float dis_spine_leg_z = 0.00f;

		pZ = trunkPosZ;
		pX -= (dis_spine_leg_x); //(boneRadius / 2.0f);
		pZ -= (spine_length + thigh_length / 2 + dis_spine_leg_z); //(0.95f * boneLength);
		createBodyi(BodyName::BodyLeftThigh, ArmLegWidth, thigh_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftThigh], pX, pY, pZ);
		createJointType(pX, pY, pZ + (thigh_length / 2), BodyName::BodySpine, BodyName::BodyLeftThigh);
		cJointID = createJointType(pX, pY, pZ + (thigh_length / 2), BodyName::BodySpine, BodyName::BodyLeftThigh, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftThigh);
		setJointID(BodyName::BodyLeftThigh, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftThigh] = BodyName::BodyTrunkLower;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftThigh)], unsigned long(0x0010));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftThigh)], unsigned long(0x7FDE));

		// left leg
		float leg_length = 0.39f;

		pZ -= (thigh_length / 2 + leg_length / 2);
		createBodyi(BodyName::BodyLeftLeg, ArmLegWidth, leg_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftLeg], pX, pY, pZ);
		cJointID = createJointType(pX, pY, pZ + (leg_length / 2), BodyName::BodyLeftThigh, BodyName::BodyLeftLeg, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftLeg);
		setJointID(BodyName::BodyLeftLeg, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftLeg] = BodyName::BodyLeftThigh;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftLeg)], unsigned long(0x0020));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftLeg)], unsigned long(0x7FAF));

		// left foot end point
		float feet_length = 0.08f + 0.127f + 0.05f;

		pZ -= (leg_length / 2 + ArmLegWidth / 4.0f);
		pY += (feet_length / 2 - ArmLegWidth / 2.0f);
		createBodyi(BodyName::BodyLeftFoot, ArmLegWidth * 0.9f, feet_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftFoot], pX, pY, pZ);
		//createJointType(pX, pY - (feet_length / 2), pZ + ArmLegWidth / 1.5f, BodyName::BodyLeftLeg, BodyName::BodyLeftFoot);
		//cJointID = createJointType(pX, pY - (feet_length / 2), pZ + ArmLegWidth / 1.5f, BodyName::BodyLeftLeg, BodyName::BodyLeftFoot, JointType::ThreeDegreeJoint);
		cJointID = createJointType(pX, trunkPosY, pZ + ArmLegWidth / 4.0f, BodyName::BodyLeftLeg, BodyName::BodyLeftFoot, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftFoot);
		setJointID(BodyName::BodyLeftFoot, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftFoot] = BodyName::BodyLeftLeg;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftFoot)], unsigned long(0x0040));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftFoot)], unsigned long(0x7FDF));

		// right thigh
		pX = trunkPosX;
		pY = trunkPosY;
		pZ = trunkPosZ;

		pX += (dis_spine_leg_x);
		pZ -= (spine_length + thigh_length / 2 + dis_spine_leg_z);
		createBodyi(BodyName::BodyRightThigh, ArmLegWidth, thigh_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightThigh], pX, pY, pZ);
		createJointType(pX, pY, pZ + (thigh_length / 2.0f), BodyName::BodySpine, BodyName::BodyRightThigh);
		cJointID = createJointType(pX, pY, pZ + (thigh_length / 2.0f), BodyName::BodySpine, BodyName::BodyRightThigh, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightThigh);
		setJointID(BodyName::BodyRightThigh, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightThigh] = BodyName::BodyTrunkLower;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightThigh)], unsigned long(0x0002));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightThigh)], unsigned long(0x7FFA));

		// right leg
		pZ -= (thigh_length / 2 + leg_length / 2);
		createBodyi(BodyName::BodyRightLeg, ArmLegWidth, leg_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightLeg], pX, pY, pZ);
		cJointID = createJointType(pX, pY, pZ + (leg_length / 2.0f), BodyName::BodyRightThigh, BodyName::BodyRightLeg, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightLeg);
		setJointID(BodyName::BodyRightLeg, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightLeg] = BodyName::BodyRightThigh;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightLeg)], unsigned long(0x0004));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightLeg)], unsigned long(0x7FF5));

		// right foot end point
		pZ -= (leg_length / 2 + ArmLegWidth / 4.0f);
		pY += (feet_length / 2 - ArmLegWidth / 2);
		createBodyi(BodyName::BodyRightFoot, ArmLegWidth * 0.9f, feet_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightFoot], pX, pY, pZ);
		//createJointType(pX, pY - (feet_length / 2), pZ + ArmLegWidth / 1.5f, BodyName::BodyRightLeg, BodyName::BodyRightFoot);
		//cJointID = createJointType(pX, pY - (feet_length / 2), pZ + ArmLegWidth / 1.5f, BodyName::BodyRightLeg, BodyName::BodyRightFoot, JointType::ThreeDegreeJoint);
		cJointID = createJointType(pX, trunkPosY, pZ + ArmLegWidth / 4.0f, BodyName::BodyRightLeg, BodyName::BodyRightFoot, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightFoot);
		setJointID(BodyName::BodyRightFoot, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightFoot] = BodyName::BodyRightLeg;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightFoot)], unsigned long(0x0008));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightFoot)], unsigned long(0x7FFB));

		float handsWidth = 0.9f * ArmLegWidth;

		// left shoulder
		float handShortenAmount=0.025f;
		float dis_spine_shoulder_x = 0.06f + 0.123f;
		float dis_spine_shoulder_z = trunk_length - (0.1363f + 0.101f);

		float shoulder_length = 0.276f+handShortenAmount;

		pX = trunkPosX;
		pY = trunkPosY;
		pZ = trunkPosZ;

		pX -= (shoulder_length / 2.0f + dis_spine_shoulder_x);
		pZ += (trunk_length - dis_spine_shoulder_z);
		createBodyi(BodyName::BodyLeftShoulder, shoulder_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftShoulder], pX, pY, pZ);
		createJointType(pX + (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyLeftShoulder);
		cJointID = createJointType(pX + (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyLeftShoulder, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftShoulder);
		setJointID(BodyName::BodyLeftShoulder, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftShoulder] = BodyName::BodyTrunkUpper;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyLeftShoulder)], unsigned long(0x0800));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyLeftShoulder)], unsigned long(0x6F7F));

		// left arm
		float arm_length = 0.278f;

		pX -= (shoulder_length / 2 + arm_length / 2);
		createBodyi(BodyName::BodyLeftArm, arm_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftArm], pX, pY, pZ);
		cJointID = createJointType(pX + (arm_length / 2.0f), pY, pZ, BodyName::BodyLeftShoulder, BodyName::BodyLeftArm, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftArm);
		setJointID(BodyName::BodyLeftArm, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftArm] = BodyName::BodyLeftShoulder;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyLeftArm], unsigned long(0x1000));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyLeftArm], unsigned long(0x57FF));

		// left hand end point
		float hand_length = 0.1f + 0.05f + 0.03f + 0.025f - handShortenAmount;

		pX -= (arm_length / 2 + (hand_length / 2.0f));
		createBodyi(BodyName::BodyLeftHand, hand_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyLeftHand], pX, pY, pZ);
		createJointType(pX + (hand_length / 2.0f), pY, pZ, BodyName::BodyLeftArm, BodyName::BodyLeftHand);
		cJointID = createJointType(pX + (hand_length / 2.0f), pY, pZ, BodyName::BodyLeftArm, BodyName::BodyLeftHand, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyLeftHand);
		setJointID(BodyName::BodyLeftHand, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyLeftHand] = BodyName::BodyLeftArm;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyLeftHand], unsigned long(0x2000));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyLeftHand], unsigned long(0x6FFF));

		// right shoulder
		pX = trunkPosX;
		pZ = trunkPosZ;

		pX += (shoulder_length / 2.0f + dis_spine_shoulder_x);
		pZ += (trunk_length - dis_spine_shoulder_z);
		createBodyi(BodyName::BodyRightShoulder, shoulder_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightShoulder], pX, pY, pZ);
		createJointType(pX - (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyRightShoulder);
		cJointID = createJointType(pX - (shoulder_length / 2.0f), pY, pZ, BodyName::BodyTrunk, BodyName::BodyRightShoulder, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightShoulder);
		setJointID(BodyName::BodyRightShoulder, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightShoulder] = BodyName::BodyTrunkUpper;

		odeGeomSetCategoryBits(mGeomID[int(BodyName::BodyRightShoulder)], unsigned long(0x0100));
		odeGeomSetCollideBits (mGeomID[int(BodyName::BodyRightShoulder)], unsigned long(0x7D7F));

		// right arm
		pX += (shoulder_length / 2 + arm_length / 2);
		createBodyi(BodyName::BodyRightArm, arm_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightArm], pX, pY, pZ);
		cJointID = createJointType(pX - (arm_length / 2.0f), pY, pZ, BodyName::BodyRightShoulder, BodyName::BodyRightArm, JointType::OneDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightArm);
		setJointID(BodyName::BodyRightArm, cJointID, JointType::OneDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightArm] = BodyName::BodyRightShoulder;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyRightArm], unsigned long(0x0200));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyRightArm], unsigned long(0x7AFF));

		// right hand end point
		pX += (arm_length / 2 + (hand_length / 2.0f));
		createBodyi(BodyName::BodyRightHand, hand_length, handsWidth);
		odeBodySetPosition(bodyIDs[BodyName::BodyRightHand], pX, pY, pZ);
		createJointType(pX - (hand_length / 2.0f), pY, pZ, BodyName::BodyRightArm, BodyName::BodyRightHand);
		cJointID = createJointType(pX - (hand_length / 2.0f), pY, pZ, BodyName::BodyRightArm, BodyName::BodyRightHand, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyRightHand);
		setJointID(BodyName::BodyRightHand, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyRightHand] = BodyName::BodyRightArm;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyRightHand], unsigned long(0x0400));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyRightHand], unsigned long(0x7DFF));

		// head
		float head_length = 0.21f + 0.04f;
		float dis_spine_head_z = 0.04f;

		float HeadWidth = 0.85f * boneRadius;

		pX = trunkPosX;
		pZ = trunkPosZ;

		pZ += (trunk_length + head_length / 2 + dis_spine_head_z);
		createBodyi(BodyName::BodyHead, HeadWidth, head_length);
		odeBodySetPosition(bodyIDs[BodyName::BodyHead], pX, pY, pZ);
		createJointType(pX, pY, pZ - (head_length / 2.0f), BodyName::BodyTrunk, BodyName::BodyHead);
		cJointID = createJointType(pX, pY, pZ - (head_length / 2.0f), BodyName::BodyTrunk, BodyName::BodyHead, JointType::ThreeDegreeJoint);
		cJointLimits = setAngleLimitations(cJointID, BodyName::BodyHead);
		setJointID(BodyName::BodyHead, cJointID, JointType::ThreeDegreeJoint, cJointLimits);
		fatherBodyIDs[BodyName::BodyHead] = BodyName::BodyTrunkUpper;

		odeGeomSetCategoryBits(mGeomID[BodyName::BodyHead], unsigned long(0x4000));
		odeGeomSetCollideBits (mGeomID[BodyName::BodyHead], unsigned long(0x7F7F));

		float mass = 0;
		for (int i = 0; i < BodyNUM; i++)
		{
			mass += odeBodyGetMass(bodyIDs[i]);
		}
		float scaleFactor = dmass / mass;
		for (int i = 0; i < BodyNUM; i++)
		{
			float mass_i = odeBodyGetMass(bodyIDs[i]);
			float length,radius;
			odeGeomCapsuleGetParams(mGeomID[i],radius,length);
			odeMassSetCapsuleTotal(bodyIDs[i],mass_i*scaleFactor,radius,length);
		}

		climberRadius = trunk_length + spine_length + leg_length + thigh_length + shoulder_length + arm_length;
		climberLegLegDis = 2 * (leg_length + thigh_length);
		climberHandHandDis = 2 * (shoulder_length + arm_length);

		return;
	}

	Vector3 computeCOM()
	{
		float totalMass=0;
		Vector3 result=Vector3::Zero();
		for (int i = 0; i < BodyNUM; i++)
		{
			float mass = odeBodyGetMass(bodyIDs[i]);
			Vector3 pos(odeBodyGetPosition(bodyIDs[i]));
			result+=mass*pos;
			totalMass+=mass;
		}
		return result/totalMass;
	}

	void setJointID(BodyName iBodyName, int iJointID, int iJointType, std::vector<Vector2>& iJointLimits)
	{
		jointIDs[iBodyName - 1] = iJointID;
		jointTypes[iBodyName - 1] = iJointType;

		if (iJointType == JointType::ThreeDegreeJoint)
		{
			jointIDIndex.push_back(iBodyName - 1);
			jointIDIndex.push_back(iBodyName - 1);
			jointIDIndex.push_back(iBodyName - 1);

			jointAxisIndex.push_back(0);
			jointAxisIndex.push_back(1);
			jointAxisIndex.push_back(2);

			jointLimits.push_back(iJointLimits[0]);
			jointLimits.push_back(iJointLimits[1]);
			jointLimits.push_back(iJointLimits[2]);
		}
		else
		{
			jointIDIndex.push_back(iBodyName - 1);
			jointAxisIndex.push_back(0);
			jointLimits.push_back(iJointLimits[0]);
		}
	}

	float convertToRad(float iDegree)
	{
		return iDegree * (PI / 180.0f);
	}

	std::vector<int> createBodyi(int i, float lx, float lz, BodyType bodyType = BodyType::BodyCapsule)
	{
		int bodyID = odeBodyCreate();
		if (i < BodyNUM && i >= 0)
		{
			bodyIDs[i] = bodyID;
			bodyTypes[i] = bodyType;
		}

		float m_body_size = lz;
		float m_body_width = lx;
		if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder 
			|| i == BodyName::BodyRightHand || i == BodyName::BodyLeftHand)
		{
			m_body_size = lx;
			m_body_width = lz;
		}
		if (i < BodyNUM && i >= 0)
		{
			boneSize[i] = m_body_size;
		}

		int cGeomID = -1;

		if (bodyType == BodyType::BodyBox)
		{
			odeMassSetBox(bodyID, DENSITY, m_body_width, boneRadius, m_body_size);
			cGeomID = odeCreateBox(m_body_width, boneRadius, m_body_size);
		}
		else if (bodyType == BodyType::BodyCapsule)
		{
			m_body_width *= 0.5f;
			
			if (i == BodyName::BodyLeftArm || i == BodyName::BodyLeftShoulder || i == BodyName::BodyRightArm || i == BodyName::BodyRightShoulder 
				|| i == BodyName::BodyRightHand || i == BodyName::BodyLeftHand)
			{
				dMatrix3 R;
				dRFromAxisAndAngle(R, 0, 1, 0, PI /2);
				odeBodySetRotation(bodyID, R);
			}
			if (i == BodyName::BodyRightFoot || i == BodyName::BodyLeftFoot)
			{
				dMatrix3 R;
				dRFromAxisAndAngle(R, 1, 0, 0, PI /2);
				odeBodySetRotation(bodyID, R);
			}
			odeMassSetCapsule(bodyID, DENSITY, m_body_width, m_body_size);
			cGeomID = odeCreateCapsule(0, m_body_width, m_body_size);
		}
		else
		{
			m_body_width *= 0.5;
			odeMassSetSphere(bodyID, DENSITY / 10, m_body_width);
			cGeomID = odeCreateSphere(m_body_width);
		}

		if (i < BodyNUM && i >= 0)
		{
			mGeomID[i] = cGeomID;
		}

		if (i >= 0)
		{
			odeGeomSetCollideBits (cGeomID, 1 << (i + 1)); 
			odeGeomSetCategoryBits (cGeomID, 1 << (i + 1)); 
		}

		odeGeomSetBody(cGeomID, bodyID);

		std::vector<int> ret_val;
		ret_val.push_back(bodyID);
		ret_val.push_back(cGeomID);
		if (i < BodyNUM && i >= 0)
		{
			initialRotations[i]=ode2eigenq(odeBodyGetQuaternion(bodyID));
		}

		return ret_val;
	}

	std::vector<Vector2> setAngleLimitations(int jointID, BodyName iBodyName)
	{
		float hipSwingFwd = convertToRad(130.0f);
        float hipSwingBack = convertToRad(20.0f);
        float hipSwingOutwards = convertToRad(70.0f);
        float hipSwingInwards = convertToRad(15.0f);
        float hipTwistInwards = convertToRad(15.0f);
        float hipTwistOutwards = convertToRad(45.0f);
            
		float shoulderSwingFwd = convertToRad(160.0f);
        float shoulderSwingBack = convertToRad(20.0f);
        float shoulderSwingOutwards = convertToRad(30.0f);
        float shoulderSwingInwards = convertToRad(100.0f);
        float shoulderTwistUp = convertToRad(80.0f);		//in t-pose, think of bending elbow so that hand points forward. This twist direction makes the hand go up
        float shoulderTwistDown = convertToRad(20.0f);

		float spineSwingSideways = convertToRad(20.0f);
        float spineSwingForward = convertToRad(40.0f);
        float spineSwingBack = convertToRad(10.0f);
        float spineTwist = convertToRad(30.0f);
        

		float fwd_limit = 30.0f * (PI / 180.0f);
		float tilt_limit = 10.0f * (PI / 180.0f);
		float twist_limit = 45.0f * (PI / 180.0f);
		/*float wristSwingFwd = 15.0f;
        float wristSwingBack = 15.0f;
        float wristSwingOutwards = 70.0f;
        float wristSwingInwards = 15.0f;
        float wristTwistRange = 30.0f;
        float ankleSwingRange = 30.0f;
        float kneeSwingRange = 140.0f;*/

		std::vector<Vector2> cJointLimits;
		const float elbowStraightLimit=AaltoGames::deg2rad*1.0f;
		const float kneeStraightLimit=AaltoGames::deg2rad*2.0f;
		const float elbowKneeBentLimit=deg2rad*150.0f;
		switch (iBodyName)
		{
		case BodyName::BodySpine:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -spineSwingForward); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, spineSwingBack);

			cJointLimits.push_back(Vector2(-spineSwingForward,spineSwingBack));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -spineSwingSideways); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, spineSwingSideways);

			cJointLimits.push_back(Vector2(-spineSwingSideways,spineSwingSideways));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -spineTwist); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, spineTwist);

			cJointLimits.push_back(Vector2(-spineTwist,spineTwist));
			break;
		case BodyName::BodyLeftThigh:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -hipSwingOutwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, hipSwingInwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -hipTwistOutwards); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, hipTwistInwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -hipSwingFwd); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, hipSwingBack);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyRightThigh:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -hipSwingInwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, hipSwingOutwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -hipTwistInwards); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, hipTwistOutwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -hipSwingFwd); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, hipSwingBack);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyLeftShoulder:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -shoulderSwingOutwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, shoulderSwingInwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -shoulderTwistDown); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, shoulderTwistUp);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -shoulderSwingBack); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, shoulderSwingFwd);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyRightShoulder:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -shoulderSwingInwards); // z
			odeJointSetAMotorParam(jointID, dParamHiStop1, shoulderSwingOutwards);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop1),odeJointGetAMotorParam(jointID,dParamHiStop1)));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -shoulderTwistDown); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, shoulderTwistUp);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop2),odeJointGetAMotorParam(jointID,dParamHiStop2)));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -shoulderSwingFwd); // x
			odeJointSetAMotorParam(jointID, dParamHiStop3, shoulderSwingBack);

			cJointLimits.push_back(Vector2(odeJointGetAMotorParam(jointID,dParamLoStop3),odeJointGetAMotorParam(jointID,dParamHiStop3)));
			break;
		case BodyName::BodyLeftLeg:
			odeJointSetHingeParam(jointID, dParamLoStop, kneeStraightLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, elbowKneeBentLimit);

			cJointLimits.push_back(Vector2(kneeStraightLimit,elbowKneeBentLimit));
			break;
		case BodyName::BodyRightLeg:
			odeJointSetHingeParam(jointID, dParamLoStop, kneeStraightLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, elbowKneeBentLimit);

			cJointLimits.push_back(Vector2(kneeStraightLimit,elbowKneeBentLimit));
			break;
		case BodyName::BodyLeftArm:
			odeJointSetHingeParam(jointID, dParamLoStop, elbowStraightLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, elbowKneeBentLimit);

			cJointLimits.push_back(Vector2(elbowStraightLimit,elbowKneeBentLimit));
			break;
		case BodyName::BodyRightArm:
			odeJointSetHingeParam(jointID, dParamLoStop, -elbowKneeBentLimit);
			odeJointSetHingeParam(jointID, dParamHiStop, -elbowStraightLimit);

			cJointLimits.push_back(Vector2(-elbowKneeBentLimit,-elbowStraightLimit));
			break;
		case BodyName::BodyHead:
			

			odeJointSetAMotorParam(jointID, dParamLoStop1, -fwd_limit); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, fwd_limit);

			cJointLimits.push_back(Vector2(-fwd_limit,fwd_limit));

			

			odeJointSetAMotorParam(jointID, dParamLoStop2, -tilt_limit); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, tilt_limit);

			cJointLimits.push_back(Vector2(-tilt_limit,tilt_limit));

			

			odeJointSetAMotorParam(jointID, dParamLoStop3, -twist_limit); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, twist_limit);

			cJointLimits.push_back(Vector2(-twist_limit,twist_limit));
			break;
		case BodyName::BodyRightFoot:
			odeJointSetHingeParam(jointID, dParamLoStop, -15 * (PI / 180.0f));
			odeJointSetHingeParam(jointID, dParamHiStop, 45 * (PI / 180.0f));

			//odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			//odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			//odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			//odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-15 * (PI / 180.0f),45 * (PI / 180.0f)));
			break;
		case BodyName::BodyLeftHand:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));
			break;
		case BodyName::BodyRightHand:
			odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-PI / 4,PI / 4));
			break;
		case BodyName::BodyLeftFoot:
			odeJointSetHingeParam(jointID, dParamLoStop, -15 * (PI / 180.0f));
			odeJointSetHingeParam(jointID, dParamHiStop, 45 * (PI / 180.0f));
			//odeJointSetAMotorParam(jointID, dParamLoStop1, -PI / 4); // x
			//odeJointSetAMotorParam(jointID, dParamHiStop1, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 4); // y
			//odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 4);

			//cJointLimits.push_back(Vector2(-PI / 4,PI / 4));

			//odeJointSetAMotorParam(jointID, dParamLoStop3, -PI / 4); // z
			//odeJointSetAMotorParam(jointID, dParamHiStop3, PI / 4);

			cJointLimits.push_back(Vector2(-15 * (PI / 180.0f),45 * (PI / 180.0f)));
			break;
		default:
			odeJointSetAMotorParam(jointID, dParamLoStop2, -PI / 2); // y
			odeJointSetAMotorParam(jointID, dParamHiStop2, PI / 2);
		}

		return cJointLimits;
	}

	Vector3 goal_pos;

	// for drawing of Env bodies
	std::vector<int> mENVGeoms;
	std::vector<int> mENVGeomTypes;

	// variables for creating joint in hold positions
	std::vector<int> jointHoldBallIDs; // only 4 ball joint IDs

	// variable refers to hold pos index
	std::vector<int> holdPosIndex; // only 4 hold IDs

	// for drawing humanoid body
	std::vector<int> bodyIDs;
	std::vector<int> fatherBodyIDs;
	std::vector<int> mGeomID; 
	std::vector<int> bodyTypes;

	// variable for getting end point of the body i
	std::vector<float> boneSize;
	
	std::vector<int> jointIDs;
	std::vector<int> jointTypes;
	int mJointSize;
	std::vector<int> jointIDIndex;
	std::vector<int> jointAxisIndex;
	std::vector<Vector2> jointLimits;

	// for testing the humanoid climber
	std::vector<float> desiredAnglesBones; 

	int masterContext;
	int maxNumContexts; 
	int currentFreeSavingStateSlot;

	float climberRadius;
	float climberLegLegDis;
	float climberHandHandDis;

}* mContext;

#include <deque>
#include <future>
class robot_CPBP
{
public:
	ControlPBP flc;
	CMAES<float> cmaes;
	SimulationContext* mContextCPBP;
	std::vector<Eigen::VectorXf> stateFeatures;
	int masterContextID;

	Eigen::VectorXf control_init_tmp;
	Eigen::VectorXf controlMin;
	Eigen::VectorXf controlMax;
	Eigen::VectorXf controlMean;
	Eigen::VectorXf controlSd;
	Eigen::VectorXf poseMin;
	Eigen::VectorXf poseMax;
	Eigen::VectorXf controlDiffSd;
	Eigen::VectorXf controlDiffDiffSd;
	Eigen::VectorXf defaultPose;
	std::vector<std::vector<RecursiveTCBSpline> > splines;  //vector for each simulation context, each vector with the number of control params (joint velocities/poses and fmax)

	int currentLoadedSavingSlot;

public:

	Vector3 bestCOMPos;

	BipedState startState;
	BipedState resetState;
	float current_cost;
	float current_cost_control;
	enum StanceBodies {sbLeftLeg=0,sbRightLeg,sbLeftHand,sbRightHand};
	enum ControlledPoses { MiddleTrunk = 0, LeftLeg = 1, RightLeg = 2, LeftHand = 3, RightHand = 4, mHead = 6, Posture = 5 };

	enum CostComponentNames {VioateDis, Velocity, ViolateVel, Angles, cLeftLeg, cRightLeg, cLeftHand, cRightHand, cMiddleTrunkP, cMiddleTrunkD, cHead, cNaN};

	robot_CPBP(SimulationContext* iContexts, BipedState& iStartState)
	{
		currentLoadedSavingSlot = iStartState.saving_slot_state;
		startState = iStartState;
		iContexts->saveContextIn(startState);

		startState.getNewCopy(iContexts->getNextFreeSavingSlot(), iContexts->getMasterContextID());

		current_cost = 0.0f;
		current_cost_control = 0.0f;

		mContextCPBP = iContexts;
		masterContextID = mContextCPBP->getMasterContextID();

		control_init_tmp = Eigen::VectorXf::Zero(mContextCPBP->getJointSize() + fmCount);
		//controller init (motor target velocities are the controlled variables, one per joint)
		controlMin = control_init_tmp;
		controlMax = control_init_tmp;
		controlMean = control_init_tmp;
		controlSd = control_init_tmp;
		controlDiffSd = control_init_tmp;
		controlDiffDiffSd = control_init_tmp;
		poseMin=control_init_tmp;
		poseMax=control_init_tmp;
		defaultPose=control_init_tmp;

		for (int i = 0; i < mContextCPBP->getJointSize(); i++)
		{
			//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
			defaultPose[i]=iContexts->getDesMotorAngleFromID(i);
			poseMin[i]=iContexts->getJointAngleMin(i);
			poseMax[i]=iContexts->getJointAngleMax(i);
			float angleRange=poseMax[i]-poseMin[i];
			controlMin[i] = -maxSpeedRelToRange*angleRange;
			controlMax[i] = maxSpeedRelToRange*angleRange;

			controlMean[i] = 0;
			controlSd[i] = 0.5f * controlMax[i];
			controlDiffSd[i] = controlDiffSdScale * controlMax[i];		//favor small accelerations
			controlDiffDiffSd[i] = 1000.0f; //NOP
		}
		//fmax control
		for (int i = mContextCPBP->getJointSize(); i < mContextCPBP->getJointSize()+fmCount; i++)
		{
			//we are making everything relative to the rotation ranges. getBodyi() could also return the controlled body (corresponding to a BodyName enum)
			controlMin[i] = minimumForce;
			controlMax[i] = maximumForce;

			controlMean[i] = 0;
			controlSd[i] = controlMax[i];
			controlDiffSd[i] = controlDiffSdScale * controlMax[i];		//favor small accelerations
			controlDiffDiffSd[i] = 1000.0f; //NOP
		}

		setCurrentOdeContext(masterContextID);

		float temp[1000];
		int stateDim = computeStateFeatures(mContextCPBP, temp); // compute state features of startState or loaded master state
		Eigen::VectorXf stateSd(stateDim);
		for (int i = 0; i < stateDim; i++)
			stateSd[i] = 0.25f;
		float control_variation = 0.1f;

		flc.init(nTrajectories, nTimeSteps / nPhysicsPerStep, stateDim, mContextCPBP->getJointSize()+fmCount, controlMin.data()
			, controlMax.data(), controlMean.data(), controlSd.data(), controlDiffSd.data(), controlDiffDiffSd.data(), control_variation, NULL);
		flc.setParams(0.25f, 0.5f,false,0.001f);
		//for each segment, the control vector has a duration value and a control point (motor speeds and fmaxes)
		cmaes.init_with_dimension(nCMAESSegments*(1+mContextCPBP->getJointSize()+fmCount));
		cmaes.selected_samples_fraction_ = 0.5;
		cmaes.use_step_size_control_ = false ;
		cmaes.minimum_exploration_variance_=0.0005f;//*controlSd[0];
		splines.resize(contextNUM);
		for (int i=0; i<contextNUM; i++)
		{
			splines[i].resize(mContextCPBP->getJointSize()+fmCount);
		}

		for (int i = 0; i <= nTrajectories; i++)
			stateFeatures.push_back(Eigen::VectorXf::Zero(stateDim));


		/*for (unsigned int k = 0; k < iContexts->getJointSize(); k++)
		{
			float stateCost = sqrtf(iContexts->getDesMotorAngleFromID(k) - iContexts->getJointAngle(k));
		}*/
	}
	VectorXf posePriorSd[nTrajectories],posePriorMean[nTrajectories],threadControls[nTrajectories];
	void optimize_the_cost(bool advance_time, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<float>& targetAngle)
	{
		setCurrentOdeContext(masterContextID);

		int cContext = getCurrentOdeContext();

		restoreOdeState(masterContextID); // we have loaded master context state

        //Update the current state and pass it to the optimizer
		Eigen::VectorXf &stateFeatureMaster = stateFeatures[masterContextID];
		computeStateFeatures(mContextCPBP, &stateFeatureMaster[0]);
		//debug-print current state cost components
		float cStateCost = computeStateCost(mContextCPBP, sourcePos, targetPos, false, targetAngle);//true
		if (advance_time) rcPrintString("Traj. cost for controller: %f", cStateCost);

		flc.startIteration(advance_time, &stateFeatureMaster[0]);
		bool standing=mContextCPBP->getHoldBodyIDs((int)(sbLeftLeg))==-1 && mContextCPBP->getHoldBodyIDs((int)(sbRightLeg))==-1;

		for (int step = 0; step < nTimeSteps/nPhysicsPerStep; step++)
		{
			flc.startPlanningStep(step);

			for (int i = 0; i < nTrajectories; i++)
			{
				if (step == 0)
                {
                    //save the physics state: at first step, the master context is copied to every other context
                    saveOdeState(i, masterContextID);
                }
                else
				{
                    //at others than the first step, just save each context so that the resampling can branch the paths
                    saveOdeState(i, i);
				}
			}			
		
			std::deque<std::future<bool>> worker_queue;
			SimulationContext::BodyName targetDrawnLines = SimulationContext::BodyName::BodyTrunk;

			for (int t = nTrajectories - 1; t >= 0; t--)
			{
				//int trajectory_idx = t;

				//lambda to be executed in the thread of the simulation context
				auto simulate_one_step = [&](int trajectory_idx)
				{
					//int c_trajectoryIdx = trajectory_idx;
	
					int previousStateIdx = flc.getPreviousSampleIdx(trajectory_idx);
					setCurrentOdeContext(trajectory_idx);

					int cContext = getCurrentOdeContext();

					restoreOdeState(previousStateIdx);

					//compute pose prior, needed for getControl()
					int nPoseParams=mContextCPBP->getJointSize();
					posePriorMean[trajectory_idx].resize(nPoseParams+fmCount);
					posePriorSd[trajectory_idx].resize(nPoseParams+fmCount);
					float dt=timeStep*(float)nPhysicsPerStep;
					posePriorMean[trajectory_idx].setZero();
					if (!standing)
					{
						for (int i = 0; i < nPoseParams; i++)
						{
							posePriorMean[trajectory_idx][i] = (defaultPose[i]-mContextCPBP->getJointAngle(i))/dt;
						}
					}
					else
					{
						for (int i = 0; i < nPoseParams; i++)
						{
							posePriorMean[trajectory_idx][i] = (0-mContextCPBP->getJointAngle(i))/dt; //t-pose: zero angles
						}
					}
					posePriorSd[trajectory_idx].head(nPoseParams)=(poseMax-poseMin)*(poseAngleSd/dt);
					posePriorSd[trajectory_idx].tail(fmCount).setConstant(1000.0f); //no additional prior on FMax
					Eigen::VectorXf &control = threadControls[trajectory_idx];
					control.resize(nPoseParams+fmCount);
					flc.getControl(trajectory_idx, control.data(),posePriorMean[trajectory_idx].data(),posePriorSd[trajectory_idx].data());
			
					//step physics
					mContextCPBP->initialPosition[trajectory_idx] = mContextCPBP->getEndPointPosBones(targetDrawnLines);
	
					bool physicsBroken = false;
					float controlCost=0;
					for (int k = 0; k < nPhysicsPerStep && !physicsBroken; k++)
					{
						//apply control
						apply_control(mContextCPBP, control);
						physicsBroken = !stepOde(timeStep ,false); //stepOdeFast(timeStep);
						if (physicsBroken)
							break;
						controlCost += compute_control_cost(mContextCPBP);

					}

					mContextCPBP->resultPosition[trajectory_idx] = mContextCPBP->getEndPointPosBones(targetDrawnLines);
					//evaluate state cost (control cost implicitly handled inside C-PBP)
					float stateCost = computeStateCost(mContextCPBP, sourcePos, targetPos,  false, targetAngle); 
					if (physicsBroken)
					{
						stateCost += 1e20;
						restoreOdeState(previousStateIdx);
					}

					Eigen::VectorXf &stateFeatureOthers = stateFeatures[trajectory_idx];
					computeStateFeatures(mContextCPBP, stateFeatureOthers.data());
					//we can add control cost to state cost, as state cost is actually uniquely associated with a tuple [previous state, control, next state]
					flc.updateResults(trajectory_idx, control.data(), stateFeatureOthers.data(), stateCost+controlCost,posePriorMean[trajectory_idx].data(),posePriorSd[trajectory_idx].data());
		
					return true;
				};
		
				worker_queue.push_back(std::async(std::launch::async,simulate_one_step,t));
			}
		
			for (std::future<bool>& is_ready : worker_queue)
			{
				is_ready.wait();
			}
		
			flc.endPlanningStep(step);
			
			//debug visualization
			debug_visualize(step);
		}
		flc.endIteration();
		cContext = getCurrentOdeContext();

		current_cost = flc.getBestTrajectoryCost();

		if (!advance_time)
		{
			//visualize
			setCurrentOdeContext(flc.getBestSampleLastIdx());
			mContextCPBP->mDrawStuff(-1,true);
			rcPrintString("Traj. cost for controller: %f", current_cost);
		}
		bestCOMPos = mContextCPBP->computeCOM();
		return;
	} 
	void clampCMAESSegmentControls(int segmentIdx,bool clampDuration, int nVelocities,int nValuesPerSegment, VectorXf &sample)
	{
		int segmentStart=segmentIdx*nValuesPerSegment;
		if (clampDuration)
		{
			sample[segmentStart]=clipMinMaxf(sample[segmentStart],minSegmentDuration,maxSegmentDuration);  
			segmentStart++;
		}
		if (cmaesSamplePoses)
		{
			for (int velIdx=0; velIdx<nVelocities; velIdx++)
			{
				sample[segmentStart+velIdx]=clipMinMaxf(sample[segmentStart+velIdx],poseMin[velIdx],poseMax[velIdx]);
			}
		}
		else
		{
			for (int velIdx=0; velIdx<nVelocities; velIdx++)
			{
				sample[segmentStart+velIdx]=clipMinMaxf(sample[segmentStart+velIdx],controlMin[velIdx],controlMax[velIdx]);
			}
		}
		for (int fmaxIdx=0; fmaxIdx<fmCount; fmaxIdx++)
		{
			//safest to start with high fmax and let the optimizer decrease them later, thus the mean at maximumForce
			sample[segmentStart+nVelocities+fmaxIdx]=clipMinMaxf(sample[segmentStart+nVelocities+fmaxIdx],minimumForce,maximumForce);
		}
	}
	CMAESTrajectoryResults cmaesResults[nTrajectories];
	CMAESTrajectoryResults bestCmaesTrajectory;
	//int bestCmaEsTrajectoryIdx;
	std::vector<std::pair<VectorXf,float> > cmaesSamples;
	void optimize_the_cost_cmaes(bool firstIter, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos, std::vector<float>& targetAngle)
	{
		setCurrentOdeContext(masterContextID);

		int cContext = getCurrentOdeContext();

		restoreOdeState(masterContextID); // we have loaded master context state
		int nVelocities=mContextCPBP->getJointSize();
		int nValuesPerSegment=1+nVelocities+fmCount;
		int nCurrentTrajectories=firstIter ? nTrajectories : nTrajectories/4;
		cmaesSamples.resize(nCurrentTrajectories);

		//at first iteration, sample the population from the control prior
		if (firstIter)
		{
			bestCmaesTrajectory.cost=FLT_MAX;
			for (int sampleIdx=0; sampleIdx<nCurrentTrajectories; sampleIdx++)
			{
				VectorXf &sample=cmaesSamples[sampleIdx].first;
				sample.resize(nCMAESSegments*nValuesPerSegment);
				for (int segmentIdx=0; segmentIdx<nCMAESSegments; segmentIdx++)
				{
					int segmentStart=segmentIdx*nValuesPerSegment;
					sample[segmentStart]=minSegmentDuration+(maxSegmentDuration-minSegmentDuration)*randomf(); //duration in 0.2...1 seconds
					if (cmaesSamplePoses)
					{
						for (int velIdx=0; velIdx<nVelocities; velIdx++)
						{
							sample[segmentStart+1+velIdx]=randGaussianClipped(defaultPose[velIdx],poseAngleSd,poseMin[velIdx],poseMax[velIdx]);
						}
					}
					else
					{
						for (int velIdx=0; velIdx<nVelocities; velIdx++)
						{
							sample[segmentStart+1+velIdx]=randGaussianClipped(controlMean[velIdx],controlSd[velIdx],controlMin[velIdx],controlMax[velIdx]);
						}
					}
					for (int fmaxIdx=0; fmaxIdx<fmCount; fmaxIdx++)
					{
						//safest to start with high fmax and let the optimizer decrease them later, thus the mean at maximumForce
						sample[segmentStart+1+nVelocities+fmaxIdx]=randGaussianClipped(maximumForce,maximumForce-minimumForce,minimumForce,maximumForce);
					}
				}
			}
		}

		//at subsequent iterations, CMAES does the sampling. Note that we clamp the samples to bounds  
		else
		{
			cmaesSamples.resize(nCurrentTrajectories);
			auto points = cmaes.sample(nCurrentTrajectories);
			//cmaesSamples[0]=cmaesSamples[bestCmaEsTrajectoryIdx];
			for (int sampleIdx=0; sampleIdx<nCurrentTrajectories; sampleIdx++)
			{
				VectorXf &sample=cmaesSamples[sampleIdx].first;
				sample=points[sampleIdx];
				//clip the values
				for (int segmentIdx=0; segmentIdx<nCMAESSegments; segmentIdx++)
				{
					clampCMAESSegmentControls(segmentIdx,true,nVelocities,nValuesPerSegment,sample);
				}
			}
		}

		//evaluate the samples, i.e., simulate in parallel and accumulate cost
		std::deque<std::future<void>> worker_queue;
		for (int sampleIdx=0; sampleIdx<nCurrentTrajectories; sampleIdx++)
		{
			VectorXf &sample=cmaesSamples[sampleIdx].first;
			float &cost=cmaesSamples[sampleIdx].second;
			cost=0;
			cmaesResults[sampleIdx].nSteps=0;
			//auto simulate_sample = [&sample,nValuesPerSegment](int sampleIdx)
			auto simulate_sample =[&](int trajectory_idx)
			{
				//restore physics state from master (all simulated trajectories start from current state)
				cmaesResults[trajectory_idx].nSteps=0;
				setCurrentOdeContext(trajectory_idx);
				restoreOdeState(masterContextID);

				//setup spline interpolation initial state
				std::vector<RecursiveTCBSpline> &spl=splines[trajectory_idx]; //this context's spline interpolators
				for (int i=0; i<nVelocities; i++)
				{
					if (cmaesSamplePoses)
					{
						spl[i].setValueAndTangent(mContextCPBP->getJointAngle(i),mContextCPBP->getJointAngleRate(i));
						spl[i].linearMix=cmaesLinearInterpolation ? 1 : 0;
					}
					else
					{
						spl[i].setValueAndTangent(mContextCPBP->getJointAngleRate(i),0);
						spl[i].linearMix=cmaesLinearInterpolation ? 1 : 0; 
					}
				}
				for (int i=nVelocities; i<nVelocities+fmCount; i++)
				{
					spl[i].setValueAndTangent(mContextCPBP->getJointFMax(i-nVelocities),0);  
					spl[i].linearMix=cmaesLinearInterpolation ? 1 : 0;
				}

				//setup spline interpolation control points: we start at segment 0, and keep the times of the next two control points in t1 and t2
				int segmentIdx=0;
				float t1=sample[segmentIdx*nValuesPerSegment];
				float t2=t1+sample[std::min(nCMAESSegments-1,segmentIdx+1)*nValuesPerSegment];
				int lastSegment=nCMAESSegments-1;
				if (!cmaesLinearInterpolation)
					lastSegment--; //in full spline interpolation, the last segment only defines the ending tangent
				float totalTime=0;
				while (segmentIdx<=lastSegment && totalTime<maxFullTrajectoryDuration)
				{					
					//interpolate
					const VectorXf &controlPoint1=sample.segment(segmentIdx*nValuesPerSegment+1,nVelocities+fmCount);
					const VectorXf &controlPoint2=sample.segment(std::min(nCMAESSegments-1,segmentIdx+1)*nValuesPerSegment+1,nVelocities+fmCount);
					VectorXf &interpolatedControl=cmaesResults[trajectory_idx].control[cmaesResults[trajectory_idx].nSteps++];
					interpolatedControl.resize(nVelocities+fmCount);
					for (int i=0; i<nVelocities; i++)
					{
						float p1=controlPoint1[i],p2=controlPoint2[i];
						if (segmentIdx==lastSegment && forceCMAESLastValueToZero)
						{
							p1=0;
							p2=0;
						}
						spl[i].step(timeStep,p1,t1,p2,t2);
						interpolatedControl[i]=spl[i].getValue();
					}
					for (int i=nVelocities; i<nVelocities+fmCount; i++)
					{
						spl[i].step(timeStep, controlPoint1[i],t1,controlPoint2[i],t2);
						interpolatedControl[i]=spl[i].getValue();
					}

					//clamp for safety
					clampCMAESSegmentControls(0,false,nVelocities,nValuesPerSegment,interpolatedControl);

					//advance spline time, and start a new segment if needed
					t1-=timeStep;
					t2-=timeStep;
					totalTime+=timeStep;
					if (t1<0.001f)
					{
						segmentIdx++;
						t1=sample[std::min(nCMAESSegments-1,segmentIdx)*nValuesPerSegment];
						t2=t1+sample[std::min(nCMAESSegments-1,segmentIdx+1)*nValuesPerSegment];
					}

					//apply the interpolated control and step forward
					apply_control_cmaes(mContextCPBP, interpolatedControl.data());
					bool physicsBroken = !stepOde(timeStep ,false); //stepOdeFast(timeStep);
					float stateCost=0;
					if (physicsBroken)
					{
						stateCost = 1e20;
						restoreOdeState(masterContextID);
					}
					else
					{
						//compute state cost, only including the hold costs at the last step
						stateCost = computeStateCost(mContextCPBP, sourcePos, targetPos,  false, targetAngle,segmentIdx==nCMAESSegments);
					}
					//evaluate state cost (control cost implicitly handled inside C-PBP)
					float controlCost = compute_control_cost(mContextCPBP);
					cost+=stateCost+controlCost;				
				} //for each step in segment
			};  //lambda for simulating sample
			if (useThreads)
				worker_queue.push_back(std::async(std::launch::async,simulate_sample,sampleIdx));
			else
				simulate_sample(sampleIdx);
		} //for each cample
		if (useThreads)
		{
			for (std::future<void>& is_ready : worker_queue)
			{
				is_ready.wait();
			}
		}

		//find min cost, convert costs to goodnesses through negating, and update cmaes
		float minCost=FLT_MAX;
		int bestIdx=0;
		for (int sampleIdx=0; sampleIdx<nCurrentTrajectories; sampleIdx++)
		{
			cmaesResults[sampleIdx].cost=cmaesSamples[sampleIdx].second;
			if (cmaesSamples[sampleIdx].second<minCost)
			{
				bestIdx=sampleIdx;
				minCost=cmaesSamples[sampleIdx].second;
			}
			cmaesSamples[sampleIdx].second*=-1.0f;
		}
		current_cost = minCost;

		rcPrintString("Traj. cost for controller: %f", current_cost);

		//Remember if this iteration produced the new best one. This is needed just in case CMAES loses it in the next iteration.
		//For example, at the end of the optimization, a reaching hand might fluctuate in and out of the hold, and the results will be rejected
		//if the hand is not reaching the hold when iteration stopped
		if (firstIter || minCost<bestCmaesTrajectory.cost)
		{
			bestCmaesTrajectory=cmaesResults[bestIdx];
		}

		//bestCmaEsTrajectoryIdx=bestIdx;
		cmaes.update(cmaesSamples,false);

		//visualize
		setCurrentOdeContext(bestIdx);
		mContextCPBP->mDrawStuff(-1,true);
		bestCOMPos = mContextCPBP->computeCOM();
	} 


	bool advance_simulation(int cTimeStep, bool debugPrint, bool flagSaveSlots, std::vector<BipedState>& nStates)
	{
		setCurrentOdeContext(masterContextID);
		restoreOdeState(masterContextID);
		int cContext = getCurrentOdeContext();
		
		Eigen::VectorXf bControl;
		float bestCost=0;
		if (optimizerType==otCPBP)
		{
			bControl = getBestControl(cTimeStep);
			bestCost = flc.getBestTrajectoryCost();
		}
		else
		{
			bControl=bestCmaesTrajectory.control[cTimeStep];
			bestCost=bestCmaesTrajectory.cost;
		}
		if (bestCost < 1e20 && bestCost > 0)
		{
			bool physicsBroken = false;
			float controlCost = 0.0f;
			for (int k = 0; k < nPhysicsPerStep && !physicsBroken; k++)
			{
				//apply control
				if (optimizerType==otCPBP)
					apply_control(mContextCPBP, bControl);
				else
					apply_control_cmaes(mContextCPBP, bControl.data());

				physicsBroken = !stepOde(timeStep,false);

				if (flagSaveSlots && !physicsBroken)
				{
					mContextCPBP->saveContextIn(startState);
					saveOdeState(masterContextID,masterContextID);

					BipedState nState = this->startState.getNewCopy(mContextCPBP->getNextFreeSavingSlot(), mContextCPBP->getMasterContextID());
					nStates.push_back(nState);

					controlCost += compute_control_cost(mContextCPBP);
				}
			}
			
			current_cost_control = controlCost;

			if (debugPrint) rcPrintString("Control cost for controller: %f",controlCost);

			if (!physicsBroken)
			{
				return true;
			}
		}

		restoreOdeState(masterContextID);
		return false;
	} 

	bool advance_simulation(bool debugPrint = false)
	{
		setCurrentOdeContext(masterContextID);
		restoreOdeState(masterContextID);
		int cContext = getCurrentOdeContext();
		/*if (optimizerType==otCMAES)
		{
			Debug::throwError("Not supported!");
		}*/
		Eigen::VectorXf bControl = getBestControl(0);
		float bestCost = flc.getBestTrajectoryCost();
		if (bestCost < 1e20 && bestCost > 0)
		{
			bool physicsBroken = false;
			float controlCost = 0.0f;
			for (int k = 0; k < nPhysicsPerStep && !physicsBroken; k++)
			{
				//apply control
				apply_control(mContextCPBP, bControl);
				physicsBroken = !stepOde(timeStep,false);

				if (!physicsBroken)
				{
					mContextCPBP->saveContextIn(startState);
					saveOdeState(masterContextID,masterContextID);
					controlCost += compute_control_cost(mContextCPBP);
				}
			}
			
			current_cost_control = controlCost;

			if (debugPrint) rcPrintString("Control cost for controller: %f",controlCost);

			if (!physicsBroken)
			{
				return true;
			}
		}

		restoreOdeState(masterContextID);
		return false;
	} 

	void takeStep()
	{
		setCurrentOdeContext(masterContextID);
		syncMasterContextWithStartState(true);
		/*for (int k = 0; k < nPhysicsPerStep; k++)
		{
			apply_zero_control(mContextCPBP);
			stepOde(timeStep,false);
		}*/
	}

	void loadPhysicsToMaster(bool loadAnyWay)
	{
		int cContext = getCurrentOdeContext();
		syncMasterContextWithStartState(loadAnyWay);
		stepOde(timeStep,false);
		restoreOdeState(masterContextID);
	}



	static float computeStateCost(SimulationContext* iContextCPBP, std::vector<ControlledPoses>& sourcePos, std::vector<Vector3>& targetPos
		,  bool printDebug, std::vector<float>& targetAngle, bool includeTargetHoldCost=true)
	{
		// we assume that we are in the correct simulation context using restoreOdeState
		float trunkDistSd = 100.0f*0.2f;  //nowadays this is only based on distance from wall
		float angleSd = poseAngleSd;  //loose prior, seems to make more natural movement
		float tPoseAngleSd=deg2rad*5.0f;


		//float postureDistSdP = 0.01f;
		//float postureDistSdA = 30.0f*(PI/180.0f);

		float endEffectorDistSd = optimizerType == otCMAES ? 0.0025f : 0.01f;  //CMAES only evaluates hold targets at last frame, need a larger weight
		float velSd = optimizerType==otCMAES ? 100.0f : 0.5f;  //velSd only needed for C-PBP to reduce noise
		float chestDirSd = 0.5f;

		float stateCost = 0;
		float preStateCost = 0;

		if (iContextCPBP->checkViolatingRelativeDis())
		{
			stateCost += 1e20;
		}
		if (printDebug)
		{
			rcPrintString("Distance violation cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		for (unsigned int k = 0; k < BodyNUM; k++)
		{
			stateCost += iContextCPBP->getBoneLinearVelocity(k).squaredNorm()/(velSd*velSd);
		}
		if (printDebug)
		{
			rcPrintString("Velocity cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		if (targetAngle.size() == 0)
		{
			if (iContextCPBP->getHoldBodyIDs((int)(ControlledPoses::RightHand - ControlledPoses::LeftLeg)) != -1
				|| iContextCPBP->getHoldBodyIDs((int)(ControlledPoses::LeftHand - ControlledPoses::LeftLeg)) != -1)
			{
				if (optimizerType==otCMAES )  //in C-PBP, pose included in the sampling proposal. 
				{
					for (int k = 0; k < iContextCPBP->getJointSize(); k++)
					{
						float diff_ang = iContextCPBP->getDesMotorAngleFromID(k) - iContextCPBP->getJointAngle(k);

						stateCost += (squared(diff_ang) /(angleSd*angleSd));
					}
				}
			}
			else
			{
				for (unsigned int k = 0; k < iContextCPBP->bodyIDs.size(); k++)
				{
					if (k!=SimulationContext::BodyLeftShoulder
						&& k!=SimulationContext::BodyLeftArm
						&& k!=SimulationContext::BodyLeftHand
						&& k!=SimulationContext::BodyRightShoulder
						&& k!=SimulationContext::BodyRightArm
						&& k!=SimulationContext::BodyRightHand)
					{
						Eigen::Quaternionf q=ode2eigenq(odeBodyGetQuaternion(iContextCPBP->bodyIDs[k]));
						float diff=q.angularDistance(initialRotations[k]);
						stateCost += squared(diff) /squared(tPoseAngleSd);
					}
				}
			} 
	
		}
		else
		{
			if (optimizerType==otCMAES)  //in C-PBP, pose included in the sampling proposal
			{
				for (int k = 0; k < iContextCPBP->getJointSize(); k++)
				{
					float diff_ang = targetAngle[k] - iContextCPBP->getJointAngle(k);

					stateCost += (squared(diff_ang) /(angleSd*angleSd));
				}
			}
		}
		if (printDebug)
		{
			rcPrintString("Pose cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		//for (unsigned int k = 0; k < BodyNUM; k++)
		//{
		//	if (iContextCPBP->getBoneAngularVelocity(k).norm() > maxSpeed)
		//	{
		//		stateCost += 2 * 1e20;
		//	}
		//}
		/*
		if (flag_do_it)
		{
			costs_components.push_back(stateCost - preStateCost);
			costs_componentsNames.push_back(CostComponentNames::ViolateVel);
			preStateCost = stateCost;
		}*/
		if (includeTargetHoldCost)
		{
			for (unsigned int i = 0; i < sourcePos.size(); i++)
			{
				ControlledPoses posID = sourcePos[i];
				Vector3 dPos = targetPos[i];
				Vector3 cPos(0.0f, 0.0f, 0.0f);

				float cWeight = 1.0f;
				//printf("debug: %d \n", i);
				switch (posID)
				{
				case robot_CPBP::MiddleTrunk:
					//cPos = iContextCPBP->getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
					cPos = iContextCPBP->computeCOM();
					////dPos[1] = cPos[1]; // y does not matter
					dPos[0] = cPos[1]; dPos[2] = cPos[2];  //only distance from wall matters
					cWeight = 1 / trunkDistSd; //weight_average_important;
					break;
				case robot_CPBP::LeftLeg:
				case robot_CPBP::RightLeg:
				case robot_CPBP::LeftHand:
				case robot_CPBP::RightHand:
					cPos = iContextCPBP->getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + posID - ControlledPoses::LeftLeg);
					cWeight = 1.0f / endEffectorDistSd;//weight_very_important;

					if (printDebug)
					{
						drawCross(cPos);
					}

					if (iContextCPBP->getHoldBodyIDs((int)(posID - ControlledPoses::LeftLeg)) != -1)// ((cPos - dPos).norm() < 0.25f * holdSize)//
					{
						cPos = dPos;
					}

					// penetrating the wall
					if (cPos.y() >= 0.15f)
					{
						stateCost = 1e20;
					}
					break;
				case robot_CPBP::mHead:
					cWeight = 0.0f; //weight_average_important / 4;// weight_not_important;
					break;
				case robot_CPBP::Posture:
					cWeight = 0.0f;
					break;
				default:
					break;
				}

				stateCost += (cWeight * (cPos - dPos)).squaredNorm();

				if (printDebug)
				{
					if ((int)posID == ControlledPoses::LeftLeg)
					{
						rcPrintString("Left leg cost %f",stateCost - preStateCost);
					}
					else if ((int)posID == ControlledPoses::RightLeg)
					{
						rcPrintString("Right leg cost %f",stateCost - preStateCost);
					}
					else if ((int)posID == ControlledPoses::LeftHand)
					{
						rcPrintString("Left hand cost %f",stateCost - preStateCost);
					}
					else if ((int)posID == ControlledPoses::RightHand)
					{
						rcPrintString("Right hand cost %f",stateCost - preStateCost);
					}
					else if ((int)posID == ControlledPoses::MiddleTrunk)
					{
						rcPrintString("Torso cost %f",stateCost - preStateCost);
					}
					preStateCost = stateCost;
				}
			}
		} //if including target hold costs

		Vector3 dirTrunk = iContextCPBP->getBodyDirectionY(SimulationContext::BodyName::BodyTrunk);
		stateCost += ((dirTrunk - Vector3(0,1,0))/chestDirSd).squaredNorm(); // chest toward the wall
		if (printDebug)
		{
			rcPrintString("Chest direction cost %f",stateCost - preStateCost);
			preStateCost = stateCost;
		}

		if (stateCost != stateCost)
		{
			stateCost = 1e20;
		}

		return stateCost;
	} 

	void syncMasterContextWithStartState(bool loadAnyWay)
	{
		int cOdeContext = getCurrentOdeContext();

		setCurrentOdeContext(ALLTHREADS);

		bool flag_set_state = false;
		for (unsigned int i = 0; i < startState.hold_bodies_ids.size(); i++)
		{
			if (startState.hold_bodies_ids[i] != mContextCPBP->getHoldBodyIDs(i))
			{
				mContextCPBP->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i));
				flag_set_state = true;
			}
		}

		bool flag_is_state_sync = false;
		if (currentLoadedSavingSlot != startState.saving_slot_state || loadAnyWay)
		{
			restoreOdeState(startState.saving_slot_state, false);
			saveOdeState(masterContextID,0);
			mContextCPBP->saveContextIn(startState);
			currentLoadedSavingSlot = startState.saving_slot_state;
			flag_is_state_sync = true;
		}

		if (flag_set_state && !flag_is_state_sync)
		{
			restoreOdeState(masterContextID, false);
			saveOdeState(masterContextID,0);
			mContextCPBP->saveContextIn(startState);
			flag_is_state_sync = true;
		}
		
		for (unsigned int i = 0; i < startState.hold_bodies_ids.size(); i++)
		{
			if (startState.hold_bodies_ids[i] != mContextCPBP->getHoldBodyIDs(i))
			{
				mContextCPBP->detachContactPoint((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i));
				if (startState.hold_bodies_ids[i] >= 0)
				{
					mContextCPBP->attachContactPointToHold((SimulationContext::ContactPoints)(SimulationContext::ContactPoints::LeftLeg + i), startState.hold_bodies_ids[i]);
				}
			}
		}

		setCurrentOdeContext(cOdeContext);

		return;
	}

	Eigen::VectorXf getBestControl(int cTimeStep)
	{
		Eigen::VectorXf control = control_init_tmp;

		flc.getBestControl(cTimeStep, control.data());
		return control;
	}

	void debug_visualize(int step)
	{
//		dsMyPushMatrix();
//		dsMyRotateZ(90.0f);

		Vector3 red = Vector3(0, 0, 255.0f) / 255.0f;
		Vector3 green = Vector3(0, 255.0f, 0) / 255.0f;
		Vector3 cyan = Vector3(255.0f, 255.0f, 0) / 255.0f;

		//for (int t = nTrajectories - 1; t >= 0; t--)
		//{
		//	int idx = t;
		//	//if (idx<=1)
		//	//	idx=1-idx;	//to visualize trajectory 1 on top of trajectory 0
		//	Vector3 color = Vector3(255, 255, 255) / 255.0f;
		//	if (flc.marginals[step + 1][t].particleRole == ParticleRole::OLD_BEST){
		//		color = green;
		//	}
		//	else if (flc.marginals[step + 1][t].particleRole == ParticleRole::POLICY)
		//	{
		//		color = cyan;
		//	}
		//	else if (flc.marginals[step + 1][t].particleRole == ParticleRole::VALIDATION){
		//		color = red;
		//	}
		//	dsSetColor(color.x(), color.y(), color.z());
		//	float p1[] = {mContextCPBP->initialPosition[idx].x(), mContextCPBP->initialPosition[idx].y(), mContextCPBP->initialPosition[idx].z()};
		//	float p2[] = {mContextCPBP->resultPosition[idx].x(), mContextCPBP->resultPosition[idx].y(), mContextCPBP->resultPosition[idx].z()};
		//	dsDrawLine(p1, p2);
		//}
		for (int t = nTrajectories - 1; t >= 0; t--)
		{
			int idx = t;
			//if (idx<=1)
			//	idx=1-idx;	//to visualize trajectory 1 on top of trajectory 0
			Vector3 color = Vector3(0.5f, 0.5f,0.5f);
			if (t==0)
				color=green;
//			if (flc.marginals[step + 1][t].particleRole == ParticleRole::POLICY)
				//color = cyan;
				rcSetColor(color.x(), color.y(), color.z());
				float p1[] = {mContextCPBP->initialPosition[idx].x(), mContextCPBP->initialPosition[idx].y(), mContextCPBP->initialPosition[idx].z()};
				float p2[] = {mContextCPBP->resultPosition[idx].x(), mContextCPBP->resultPosition[idx].y(), mContextCPBP->resultPosition[idx].z()};
				rcDrawLine(p1, p2);
		}

//		dsMyPopMatrix();
		return;
	} 

	static void drawCross(const Vector3& p, Vector3 color = Vector3(0.0f,0.0f,1.0f))
	{
//		dsMyPushMatrix();
//		dsMyRotateZ(90.0f);

		//Vector3 color = Vector3(0, 0, 255.0f) / 255.0f;
		rcSetColor(color.x(), color.y(), color.z());

		float cross_size = 0.3f;
		float p1[] = {p.x() - cross_size / 2, p.y(), p.z()};
		float p2[] = {p.x() + cross_size / 2, p.y(), p.z()};
		rcDrawLine(p1, p2);

		p1[0] = p.x();
		p1[1] = p.y() - cross_size / 2;
		p2[0] = p.x();
		p2[1] = p.y() + cross_size / 2;
		rcDrawLine(p1, p2);

		p1[1] = p.y();
		p1[2] = p.z() - cross_size / 2;
		p2[1] = p.y();
		p2[2] = p.z() + cross_size / 2;
		rcDrawLine(p1, p2);

//		dsMyPopMatrix();
		return;
	}

	static void apply_control(SimulationContext* iContextCPBP, const Eigen::VectorXf& control)
	{
		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			float c_j = control[j];
			iContextCPBP->setMotorSpeed(j, c_j);
		}
		if (fmCount!=0)
		{
			//fmax values for each group are after the joint motor speeds
			iContextCPBP->setMotorGroupFmaxes(&control[iContextCPBP->getJointSize()]);
		}
	}
	static void apply_control_cmaes(SimulationContext* iContextCPBP, const float *control)
	{
		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			float c_j = control[j];
			if (cmaesSamplePoses)
			{
				iContextCPBP->driveMotorToPose(j,c_j);
			}
			else
			{
				iContextCPBP->setMotorSpeed(j, c_j);
			}
		}
		if (optimizerType==otCMAES && fmCount!=0)
		{
			//fmax values for each group are after the joint motor speeds
			iContextCPBP->setMotorGroupFmaxes(&control[iContextCPBP->getJointSize()]);
		}
	}
	static void apply_zero_control(SimulationContext* iContextCPBP)
	{
		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			float c_j = 0.0f;
			iContextCPBP->setMotorSpeed(j, c_j);
		}

		/*for (int j = 0; j < iContextCPBP->getHoldJointSize(); j++)
		{
			float c_j = control[iContextCPBP->getJointSize() + j];
			iContextCPBP->setMotorHoldSpeed(j, c_j);
		}*/
	}

	static float compute_control_cost(SimulationContext* iContextCPBP)
	{
		float result=0;
		for (int j = 0; j < iContextCPBP->getJointSize(); j++)
		{
			result += iContextCPBP->getMotorAppliedSqTorque(j)/squared(forceCostSd);
		}
		result+=iContextCPBP->getSqForceOnFingers()/squared(forceCostSd);
		return result;
	}



	static void pushStateFeature(int &featureIdx, float *stateFeatures, const Vector3& v)
	{
		stateFeatures[featureIdx++] = v.x();
		stateFeatures[featureIdx++] = v.y();
		stateFeatures[featureIdx++] = v.z();
	}

	static void pushStateFeature(int &featureIdx, float *stateFeatures, const Vector4& v)
	{
		stateFeatures[featureIdx++] = v.x();
		stateFeatures[featureIdx++] = v.y();
		stateFeatures[featureIdx++] = v.z();
		stateFeatures[featureIdx++] = v.w();
	}

	static void pushStateFeature(int &featureIdx, float *stateFeatures, const float& f)
	{
		stateFeatures[featureIdx++] = f;
	}

	static int computeStateFeatures(SimulationContext* iContextCPBP, float *stateFeatures) // BipedState state
	{
	
		int featureIdx = 0;
		const int nStateBones=6;
		SimulationContext::BodyName stateBones[6]={
			SimulationContext::BodySpine,
			SimulationContext::BodyTrunk,
			SimulationContext::BodyRightArm,
			SimulationContext::BodyRightLeg,
			SimulationContext::BodyLeftArm,
			SimulationContext::BodyLeftLeg};


		for (int i = 0; i < nStateBones; i++)
		{
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBonePosition(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneLinearVelocity(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneAngle(stateBones[i]));
			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getBoneAngularVelocity(stateBones[i]));
		}

//		for (int i = 0; i < iContextCPBP->getJointSize(); i++)
//		{
//			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getJointAngle(i));
////			pushStateFeature(featureIdx, stateFeatures, iContextCPBP->getJointSpeed(i));
//		}

		for (int i = 0; i < iContextCPBP->getHoldBodyIDsSize(); i++)
		{
			if (iContextCPBP->getHoldBodyIDs(i) != -1)
			{
				pushStateFeature(featureIdx, stateFeatures, 1.0f);
				//pushStateFeature(featureIdx, stateFeatures, SimulationContext::getAbsAngleBtwVectors(Vector3(0.0f,0.0f,1.0f), -iContextCPBP->getBodyDirectionZ(SimulationContext::BodyName::BodyLeftLeg + i)));
				//pushStateFeatureFloat(featureIdx, stateFeatures, c.holds_joints[i]->GetJointSpeed());
			}
			else
			{
				pushStateFeature(featureIdx, stateFeatures, 0.0f);
				//pushStateFeature(featureIdx, stateFeatures, 10.0f);
				//pushStateFeatureFloat(featureIdx, stateFeatures, 10.0f);
			}
		}
		return featureIdx;
	}

}* mCPBP;

class mSampleStructure
{
public:
	void initialization()
	{
		toNodeGraph = -1;
		fromNodeGraph = -1;

		cOptimizationCost = 0.0f;
		numItrFixedCost = 0;

		isOdeConstraintsViolated = false;
		isReached = false;
		isRejected = false;

		// variables for recovery from playing animation
		restartSampleStartState = false;

		for (int i = 0; i < 4 ; i++)
		{
			to_hold_ids.push_back(-1);
		}
		toNode = -1;

		control_cost = 0.0f;
	}

	mSampleStructure()
	{
		initialization();
		isSet = false;

		closest_node_index = -1;
	}

	mSampleStructure(std::vector<robot_CPBP::ControlledPoses>& iSourceP, std::vector<Vector3>& iDestinationP,
					 std::vector<int>& iInitialHoldIDs, std::vector<int>& iDesiredHoldIDs, int iClosestIndexNode,
					 std::vector<Vector3>& iWorkSpaceHolds, std::vector<Vector3>& iWorkSpaceColor, std::vector<Vector3>& iDPoint)
	{
		initialization();
		isSet = true;

		for (unsigned int i = 0; i < iSourceP.size(); i++)
		{
			sourceP.push_back(iSourceP[i]);
			destinationP.push_back(iDestinationP[i]);
		}

		initial_hold_ids = iInitialHoldIDs;
		desired_hold_ids = iDesiredHoldIDs;

		closest_node_index = iClosestIndexNode;

		// for debug visualization
		mWorkSpaceHolds = iWorkSpaceHolds;
		mWorkSpaceColor = iWorkSpaceColor;
		dPoint = iDPoint;
	}

	//mSampleStructure(std::vector<robot_CPBP::ControlledPoses>& iSourceP, std::vector<float>& iDestinationA, std::vector<Vector3>& iDestinationP,
	//				 std::vector<int>& iInitialHoldIDs, std::vector<int>& iDesiredHoldIDs, int iClosestIndexNode,
	//				 std::vector<Vector3>& iWorkSpaceHolds, std::vector<Vector3>& iWorkSpaceColor, std::vector<Vector3>& iDPoint)
	//{
	//	for (unsigned int i = 0; i < iSourceP.size(); i++)
	//	{
	//		sourceP.push_back(iSourceP[i]);
	//	}
	//	for (unsigned int i = 0; i < iDestinationA.size(); i++)
	//	{
	//		destinationA.push_back(iDestinationA[i]);
	//	}
	//	for (unsigned int i = 0; i < iDestinationP.size(); i++)
	//	{
	//		destinationP.push_back(iDestinationP[i]);
	//	}
	//	initial_hold_ids = iInitialHoldIDs;
	//	desired_hold_ids = iDesiredHoldIDs;
	//	cOptimizationCost = 0.0f;
	//	numItrFixedCost = 0;
	//	closest_node_index = iClosestIndexNode;
	//	isOdeConstraintsViolated = false;
	//	isReached = false;
	//	isRejected = false;
	//	isSet = true;
	//	// for debug visualization
	//	mWorkSpaceHolds = iWorkSpaceHolds;
	//	mWorkSpaceColor = iWorkSpaceColor;
	//	dPoint = iDPoint;
	//	// variables for recovery from playing animation
	//	restartSampleStartState = false;
	//	for (int i = 0; i < 4 ; i++)
	//	{
	//		to_hold_ids.push_back(-1);
	//	}
	//	toNode = -1;
	//}

	void draw_ws_points(Vector3& _from)
	{
//		dsMyPushMatrix();
//		dsMyRotateZ(90.0f);

		//for (unsigned int i = 0; i < mWorkSpaceHolds.size(); i++)
		//{
		//	Vector3 _to = mWorkSpaceHolds[i];
		//	rcSetColor(mWorkSpaceColor[i].x(), mWorkSpaceColor[i].y(), mWorkSpaceColor[i].z());
		//	float p1[] = {_from.x(), _from.y(), _from.z()};
		//	float p2[] = {_to.x(), _to.y(), _to.z()};
		//	rcDrawLine(p1, p2);
		//}

		Vector3 color(0,0,1);
		float mCubeSize = 0.1f;
		for (unsigned int i = 0; i < dPoint.size(); i++)
		{
			Vector3 mCenter = dPoint[i];
			rcSetColor(color.x(), color.y(), color.z());
			float p1[] = {mCenter.x() - mCubeSize, mCenter.y(), mCenter.z() - mCubeSize};
			float p2[] = {mCenter.x() - mCubeSize, mCenter.y(), mCenter.z() + mCubeSize};
			float p3[] = {mCenter.x() + mCubeSize, mCenter.y(), mCenter.z() + mCubeSize};
			float p4[] = {mCenter.x() + mCubeSize, mCenter.y(), mCenter.z() - mCubeSize};
			rcDrawLine(p1, p2);
			rcDrawLine(p2, p3);
			rcDrawLine(p3, p4);
			rcDrawLine(p4, p1);
		}

		//dsMyPopMatrix();
		return;
	}

	std::vector<robot_CPBP::ControlledPoses> sourceP; // contains head, trunk and contact points sources
	std::vector<Vector3> destinationP; // contains head's desired angle, and trunk's and contact points's desired positions (contact points's desired positions have the most value to us)
	std::vector<float> destinationA;

	std::vector<int> desired_hold_ids; // desired holds's ids to reach
	std::vector<int> initial_hold_ids; // connected joints to (ll,rl,lh,rh); -1 means it is disconnected, otherwise it is connected to the hold with the same id

	std::vector<int> to_hold_ids; // just for debugging
	int toNode; // from closest node to toNode
//	int seeChildOfNode; // if it was failure then see child of see of

	int toNodeGraph;
	int fromNodeGraph;

	// check cost change
	float cOptimizationCost;
	int numItrFixedCost;

	int closest_node_index;
	std::vector<BipedState> statesFromTo;
	
	bool isOdeConstraintsViolated;
	bool isReached;
	bool isRejected;
	bool isSet;

	// for debug visualization
	std::vector<Vector3> mWorkSpaceHolds;
	std::vector<Vector3> mWorkSpaceColor;
	std::vector<Vector3> dPoint;

	// variables for recovery from playing animation
	bool restartSampleStartState;

	// handling energy cost - control cost
	float control_cost;
};

class mNode
{
private:
	float costFromFatherToThis(std::vector<BipedState>& _fromFatherToThis)
	{
		float _cost = 0;
		for (int i = 0; i < (int)(_fromFatherToThis.size() - 1); i++)
		{
			_cost += getDisFrom(_fromFatherToThis[i], _fromFatherToThis[i+1]);
		}
		return _cost;
	}

	float getDisFrom(BipedState& c1, BipedState& c2) // used for calculating cost
	{
		float dis = 0.0f;
		for (unsigned int i = 0; i < c1.bodyStates.size(); i++)
		{
			BodyState m_b_i = c1.bodyStates[i];
			BodyState c_b_i = c2.bodyStates[i];
			dis += (m_b_i.getPos() - c_b_i.getPos()).squaredNorm();
		}
		return sqrtf(dis);
	}
public:
	mNode(BipedState& iState, int iFatherIndex, int iNodeIndex, std::vector<BipedState>& _fromFatherToThis)
	{
		cNodeInfo = iState;

		statesFromFatherToThis = _fromFatherToThis;

		mFatherIndex = iFatherIndex;
		nodeIndex = iNodeIndex;

		cCost = 0.0f;
		control_cost = 0.0f;

		_goalNumNodeAddedFor = -1;
	}

	bool isNodeEqualTo(std::vector<int>& s_i)
	{
		if (mNode::isSetAEqualsSetB(s_i, cNodeInfo.hold_bodies_ids))
		{
			return true;
		}
		if (mNode::isSetAEqualsSetB(s_i, poss_hold_ids))
		{
			return true;
		}
		return false;
	}

	// distance to the whole body state
	float getDisFrom(BipedState& c) // used for adding node to the tree
	{
		float dis = 0.0f;
		for (unsigned int i = 0; i < cNodeInfo.bodyStates.size(); i++)
		{
			BodyState m_b_i = cNodeInfo.bodyStates[i];
			BodyState c_b_i = c.bodyStates[i];
			dis += (m_b_i.getPos() - c_b_i.getPos()).squaredNorm();
			//dis += (m_b_i.vel - c_b_i.vel).LengthSquared();
			//dis += squared(m_b_i.angle - c_b_i.angle);
			//dis += squared(m_b_i.aVel - c_b_i.aVel);
		}
		return sqrtf(dis);
	}

	// just given middle trunk pose or hand pos
	float getSumDisEndPosTo(Vector3& iP)
	{
		float dis = 0.0f;
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg) - iP).squaredNorm();
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::RightLeg)- iP).squaredNorm();
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::LeftArm)- iP).squaredNorm();
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::RightArm)- iP).squaredNorm();
		dis += (cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk)- iP).squaredNorm();
		return sqrtf(dis);
	}

	float getCostDisconnectedArmsLegs()
	{
		int counter_disconnected_legs = 0;
		int counter_disconnected_arms = 0;
		for (unsigned int i = 0; i < cNodeInfo.hold_bodies_ids.size(); i++)
		{
			if (i <= 1 && cNodeInfo.hold_bodies_ids[i] == -1)
				counter_disconnected_legs++;
			if (i > 1 && cNodeInfo.hold_bodies_ids[i] == -1)
				counter_disconnected_arms++;
		}
		return 10.0f * (counter_disconnected_legs + counter_disconnected_arms);
	}

	float getCostNumMovedLimbs(std::vector<int>& iStance)
	{
		int mCount = getDiffBtwSetASetB(iStance, this->cNodeInfo.hold_bodies_ids);
		
		return 1.0f * mCount;
	}

	/*float getSumDisconnectedCost()
	{
		int counter_disconnected_legs = 0;
		int counter_disconnected_arms = 0;
		for (unsigned int i = 0; i < cNodeInfo.hold_bodies_ids.size(); i++)
		{
			if (i <= 1 && cNodeInfo.hold_bodies_ids[i] == -1)
				counter_disconnected_legs++;
			if (i > 1 && cNodeInfo.hold_bodies_ids[i] == -1)
				counter_disconnected_arms++;
		}
		float disconnected_cost = 0.0f;
		if (counter_disconnected_legs > 1)
		{
			disconnected_cost += 1.0f;
		}
		if (counter_disconnected_legs > 0 && counter_disconnected_arms > 0)
		{
			disconnected_cost += 0.5f;
		}
		return disconnected_cost;
	}*/

	float getSumDisEndPosTo(std::vector<int>& iIDs, std::vector<Vector3>& iP)
	{
		float dis = 0.0f;
		
		for (unsigned int i = 0; i < iIDs.size(); i++)
		{
			if (iIDs[i] != -1)
			{
				dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + i) - iP[i]).squaredNorm();
			}
		}
		
		return dis;
	}
	
	/*float getSumDisEndPosTo(mSampleStructure& mSample)
	{
		float dis = 0;
		for (unsigned int i = 0; i < mSample.sourceP.size(); i++)
		{
			switch (mSample.sourceP[i])
			{
			case robot_CPBP::ControlledPoses::LeftLeg:
				dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg) - mSample.destinationP[i]).LengthSquared();
				break;
			case robot_CPBP::ControlledPoses::RightLeg:
				dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::RightLeg) - mSample.destinationP[i]).LengthSquared();
				break;
			case robot_CPBP::ControlledPoses::LeftHand:
				dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::LeftArm) - mSample.destinationP[i]).LengthSquared();
				break;
			case robot_CPBP::ControlledPoses::RightHand:
				dis += (cNodeInfo.getEndPointPosBones(SimulationContext::ContactPoints::RightArm) - mSample.destinationP[i]).LengthSquared();
				break;
			case robot_CPBP::ControlledPoses::MiddleTrunk:
				dis += (cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk) - mSample.destinationP[i]).LengthSquared();
				break;
			}
		}
		return sqrtf(dis);
	}*/

	/*std::vector<int> getInitialBodyHolds(mSampleStructure& cSample)
	{
		std::vector<int> nHoldIds;
		for (unsigned int i = 0; i < cNodeInfo.hold_bodies_ids.size(); i++)
		{
			if (cSample.initial_hold_ids[i] != -1)
				nHoldIds.push_back(cNodeInfo.hold_bodies_ids[i]);
			else
				nHoldIds.push_back(-1);
		}

		if (nHoldIds[nHoldIds.size() - 1] == -1 && nHoldIds[nHoldIds.size() - 2] == -1)
			return std::vector<int>();
		return nHoldIds;
	}*/

	/*int getNumInContactHoldsWith(std::vector<int>& iSample_desired_hold_ids)
	{
		int sum_inContact_holds = 0;
		for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
		{
			if (cNodeInfo.hold_bodies_ids[i] == iSample_desired_hold_ids[i])
			{
				sum_inContact_holds++;
			}
		}
		return sum_inContact_holds;
	}*/

	bool addTriedHoldSet(std::vector<int>& iSample_desired_hold_ids)
	{
		if (!isInTriedHoldSet(iSample_desired_hold_ids))
		{
			std::vector<int> nSample;
			for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
			{
				nSample.push_back(iSample_desired_hold_ids[i]);
			}
			mTriedHoldSet.push_back(nSample);
			return true;
		}
		return false;
	}

	bool isInTriedHoldSet(std::vector<int>& iSample_desired_hold_ids)
	{
		for (unsigned int i = 0; i < mTriedHoldSet.size(); i++)
		{
			std::vector<int> t_i = mTriedHoldSet[i];
			bool flag_found_try = isSetAEqualsSetB(t_i, iSample_desired_hold_ids);
			if (flag_found_try)
			{
				return true;
			}
		}
		return false;
	}

	static bool isSetAGreaterThan(std::vector<int>& set_a, int f)
	{
		for (unsigned int j = 0; j < set_a.size(); j++)
		{
			if (set_a[j] <= f)
			{
				return false;
			}
		}
		return true;
	}

	static bool isSetAEqualsSetB(const std::vector<int>& set_a, const std::vector<int>& set_b)
	{
		//AALTO_ASSERT1(set_a.size()==4 && set_b.size()==4);
		if (set_a.size() != set_b.size())
			return false;

		if (set_a.size()==4 && set_b.size()==4)
		{
			int diff=0;
			diff+=abs(set_a[0]-set_b[0]);
			diff+=abs(set_a[1]-set_b[1]);
			diff+=abs(set_a[2]-set_b[2]);
			diff+=abs(set_a[3]-set_b[3]);
			return diff==0;
		}
		for (unsigned int i = 0; i < set_a.size(); i++)
		{
			if (set_a[i] != set_b[i])
			{
				return false;
			}
		}

		return true;

	}

	static int getDiffBtwSetASetB(std::vector<int>& set_a, std::vector<int>& set_b)
	{
		int mCount = 0;
		for (unsigned int i = 0; i < set_a.size(); i++)
		{
			if (set_a[i] != set_b[i])
			{
				mCount++;
			}
		}
		return mCount;
	}

	BipedState cNodeInfo;
	std::vector<BipedState> statesFromFatherToThis;

	std::vector<int> poss_hold_ids;
	// variables for building the tree
	int nodeIndex;
	int mFatherIndex; // -1 means no father or root node
	std::vector<int> mChildrenIndices; // vector is empty

	std::vector<std::vector<int>> mTriedHoldSet;

	float cCost;
	float control_cost; // handling energy consumption
	int _goalNumNodeAddedFor;
};

#include "mKDTree.h"
class mSampler
{
public:
	float climberRadius;
	float climberLegLegDis;
	float climberHandHandDis;

	KDTree<int> mHoldsKDTree;
	std::vector<Vector3> myHoldPoints;
	
	// helping sampling
	std::vector<std::vector<int>> indices_higher_than;
	std::vector<std::vector<int>> indices_lower_than;

	mSampler(SimulationContext* iContextRRT)
		:mHoldsKDTree(3)
	{
		climberRadius = iContextRRT->getClimberRadius();
		climberLegLegDis = iContextRRT->getClimberLegLegDis();
		climberHandHandDis = iContextRRT->getClimberHandHandDis();

		for (unsigned int i = 0; i < iContextRRT->holds_body.size(); i++)
		{
			Vector3 hPos = iContextRRT->holds_body[i];

			myHoldPoints.push_back(hPos);
			mHoldsKDTree.insert(getHoldKey(hPos), myHoldPoints.size() - 1);
		}

		fillInLowerHigherHoldIndices();
	}

	/////////////////////////////////////////////////////// sample costs ////////////////////////////////////////////////////////////////////
	float getSampleCost(std::vector<int>& desiredStance, std::vector<int>& sampledPriorStance)
	{
		float iDis = 0.0f;
		for (unsigned int j = 0; j < desiredStance.size(); j++)
		{
			if (sampledPriorStance[j] != -1)
			{
				iDis += (getHoldPos(desiredStance[j]) - getHoldPos(sampledPriorStance[j])).norm();
			}
			else
			{
				iDis += 10.0f;
			}
		}
		return iDis;
	}

	/*float getWeightForSample(mNode* closestNode, std::vector<int>& sample_i, std::vector<int>& iSample_desired_hold_ids)
	{
		float iDis = getSampleCost(iSample_desired_hold_ids, sample_i);
		
		return 1/(1 + fabs(iDis));
	}*/

	bool isFromStanceToStanceValid(std::vector<int>& _formStanceIds, std::vector<int>& _toStanceIds, bool isInitialStance)
	{
		if (!isAllowedHandsLegsInDSample(_formStanceIds, _toStanceIds, isInitialStance)) // are letting go of hands and legs allowed
		{
			return false;	
		}

		std::vector<Vector3> sample_n_hold_points; float size_n = 0;
		Vector3 midPointN = getHoldStancePosFrom(_toStanceIds, sample_n_hold_points, size_n);
		if (size_n == 0)
		{
			return false;
		}

		if (!acceptDirectionLegsAndHands(midPointN, _toStanceIds, sample_n_hold_points))
		{
			return false;
		}

		if (!isFromStanceCloseEnough(_formStanceIds, _toStanceIds))
		{
			return false;
		}

		if (!earlyAcceptOfSample(_toStanceIds, isInitialStance)) // is it kinematically reachable
		{
			return false;
		}
		return true;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Vector3 getHoldPos(int i)
	{
		if (myHoldPoints.size() == 0)
		{
			return Vector3(0.0f, 0.0f, 0.0f);
		}
		Vector3 rPoint = myHoldPoints[0];
		if (i >= 0)
		{
			rPoint = myHoldPoints[i];
		}
		return rPoint;
	}

	Vector3 getHoldStancePosFrom(std::vector<int>& sample_desired_hold_ids, std::vector<Vector3>& sample_desired_hold_points, float& mSize)
	{
		mSize = 0;
		sample_desired_hold_points.clear();
		Vector3 midPoint(0.0f,0.0f,0.0f);
		for (unsigned int i = 0; i < sample_desired_hold_ids.size(); i++)
		{
			if (sample_desired_hold_ids[i] != -1)
			{
				sample_desired_hold_points.push_back(getHoldPos(sample_desired_hold_ids[i]));
				midPoint += sample_desired_hold_points[i];
				mSize++;
			}
			else
				sample_desired_hold_points.push_back(Vector3(0,0,0));
		}

		if (mSize > 0)
		{
			midPoint = midPoint / mSize;
		}

		return midPoint;
	}

	std::vector<std::vector<int>> getListOfSamples(mNode* closestNode, std::vector<int>& iSample_desired_hold_ids, bool isInitialStance)
	{
		std::vector<std::vector<int>> list_samples;

		std::vector<Vector3> sample_Des_hold_points; float size_d = 0;
		Vector3 midPointD = getHoldStancePosFrom(iSample_desired_hold_ids, sample_Des_hold_points, size_d);

		std::vector<std::vector<size_t>> workSpaceHoldIDs;
		std::vector<Vector3> workspacePoints; // for debug visualization
		std::vector<Vector3> workSpacePointColor; // for debug visualization
		std::vector<bool> isIDInWorkSapceVector; // is current id in workspace or not

		getAllHoldsInRangeAroundAgent(closestNode, workSpaceHoldIDs, workspacePoints, workSpacePointColor, isIDInWorkSapceVector);

		std::vector<int> initial_holds_ids = closestNode->cNodeInfo.hold_bodies_ids;

		std::vector<int> diff_hold_index;
		std::vector<int> same_hold_index;

		for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
		{
			if (initial_holds_ids[i] != iSample_desired_hold_ids[i] || initial_holds_ids[i] == -1)
			{
				diff_hold_index.push_back(i);
			}
			else
			{
				same_hold_index.push_back(i);
			}
		}

		std::vector<std::vector<int>> possible_hold_index_diff;
		std::vector<unsigned int> itr_index_diff;
		for (unsigned int i = 0; i < diff_hold_index.size(); i++)
		{
			std::vector<int> possible_hold_diff_i;
			int index_diff_i = diff_hold_index[i];

			addToSetHoldIDs(-1, possible_hold_diff_i);
			addToSetHoldIDs(iSample_desired_hold_ids[index_diff_i], possible_hold_diff_i);
			
			if (!isInitialStance)
			{
				Vector3 cPos_i = closestNode->cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg + index_diff_i);
				int nHold_id = getClosest_WSHoldIndex_ToDesiredPos(workSpaceHoldIDs[index_diff_i], iSample_desired_hold_ids[index_diff_i], initial_holds_ids[index_diff_i], cPos_i);
				addToSetHoldIDs(nHold_id, possible_hold_diff_i);
			}

			addToSetHoldIDs(initial_holds_ids[index_diff_i], possible_hold_diff_i);

			itr_index_diff.push_back(0);
			possible_hold_index_diff.push_back(possible_hold_diff_i);
		}

		bool flag_continue = true;

		if (diff_hold_index.size() == 0)
		{
			flag_continue = false;
			list_samples.push_back(iSample_desired_hold_ids);
		}

		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n;
			for (int i = iSample_desired_hold_ids.size() - 1; i >= 0; i--)
			{
				sample_n.push_back(-1);
			}
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				int index_diff_i = diff_hold_index[i];
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[index_diff_i] = possible_hold_diff_i[itr_index_diff_i];
			}
			for (unsigned int i = 0; i < same_hold_index.size(); i++)
			{
				int index_same_i = same_hold_index[i];
				sample_n[index_same_i] = initial_holds_ids[index_same_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= possible_hold_index_diff[i].size())
				{
					if (i == diff_hold_index.size() - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////
			if (!isAllowedHandsLegsInDSample(initial_holds_ids, sample_n, isInitialStance)) // are letting go of hands and legs allowed
			{
				continue;	
			}

			/*if (!acceptEqualPart(sample_n, initial_holds_ids, iSample_desired_hold_ids))
			{
				continue;
			}*/

			std::vector<Vector3> sample_n_hold_points; float size_n = 0;
			Vector3 midPointN = getHoldStancePosFrom(sample_n, sample_n_hold_points, size_n);
			if (size_n == 0)
			{
				continue;
			}

			if (!acceptDirectionLegsAndHands(midPointN, sample_n, sample_n_hold_points))
			{
				continue;
			}

			if (!isFromStanceCloseEnough(initial_holds_ids, sample_n))
			{
				continue;
			}

			if (!earlyAcceptOfSample(sample_n, isInitialStance)) // is it kinematically reachable
			{
				continue;
			}
			if (closestNode->isInTriedHoldSet(sample_n)) // is it tried
			{
				continue;
			}

			list_samples.push_back(sample_n);
		}

		return list_samples;
	}

	std::vector<std::vector<int>> getListOfStanceSamples(std::vector<int>& from_hold_ids, std::vector<int>& to_hold_ids, bool isInitialStance)
	{
		std::vector<std::vector<int>> list_samples;

		std::vector<int> initial_holds_ids = from_hold_ids;

		std::vector<int> diff_hold_index;
		std::vector<int> same_hold_index;

		for (unsigned int i = 0; i < to_hold_ids.size(); i++)
		{
			if (initial_holds_ids[i] != to_hold_ids[i] || initial_holds_ids[i] == -1)
			{
				diff_hold_index.push_back(i);
			}
			else
			{
				same_hold_index.push_back(i);
			}
		}

		std::vector<std::vector<int>> possible_hold_index_diff;
		std::vector<unsigned int> itr_index_diff;
		for (unsigned int i = 0; i < diff_hold_index.size(); i++)
		{
			std::vector<int> possible_hold_diff_i;
			int index_diff_i = diff_hold_index[i];

			addToSetHoldIDs(-1, possible_hold_diff_i);
			addToSetHoldIDs(to_hold_ids[index_diff_i], possible_hold_diff_i);

			addToSetHoldIDs(initial_holds_ids[index_diff_i], possible_hold_diff_i);

			itr_index_diff.push_back(0);
			possible_hold_index_diff.push_back(possible_hold_diff_i);
		}

		bool flag_continue = true;

		if (diff_hold_index.size() == 0)
		{
			flag_continue = false;
			list_samples.push_back(to_hold_ids);
		}

		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n;
			for (int i = to_hold_ids.size() - 1; i >= 0; i--)
			{
				sample_n.push_back(-1);
			}
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				int index_diff_i = diff_hold_index[i];
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[index_diff_i] = possible_hold_diff_i[itr_index_diff_i];
			}
			for (unsigned int i = 0; i < same_hold_index.size(); i++)
			{
				int index_same_i = same_hold_index[i];
				sample_n[index_same_i] = initial_holds_ids[index_same_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < diff_hold_index.size(); i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= possible_hold_index_diff[i].size())
				{
					if (i == diff_hold_index.size() - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////
			if (!isAllowedHandsLegsInDSample(initial_holds_ids, sample_n, isInitialStance)) // are letting go of hands and legs allowed
			{
				continue;	
			}

			std::vector<Vector3> sample_n_hold_points; float size_n = 0;
			Vector3 midPointN = getHoldStancePosFrom(sample_n, sample_n_hold_points, size_n);
			if (size_n == 0)
			{
				continue;
			}

			if (!acceptDirectionLegsAndHands(midPointN, sample_n, sample_n_hold_points))
			{
				continue;
			}

			if (!isFromStanceCloseEnough(initial_holds_ids, sample_n))
			{
				continue;
			}

			if (!earlyAcceptOfSample(sample_n, isInitialStance)) // is it kinematically reachable
			{
				continue;
			}

			list_samples.push_back(sample_n);
		}

		return list_samples;
	}

	void getListOfStanceSamplesAround(int to_hold_id, std::vector<int>& from_hold_ids, std::vector<std::vector<int>> &out_list_samples)
	{
		/*std::vector<int> dStance;
		dStance.push_back(1);
		dStance.push_back(1);
		dStance.push_back(2);
		dStance.push_back(3);

		std::vector<int> fStance;
		fStance.push_back(0);
		fStance.push_back(1);
		fStance.push_back(2);
		fStance.push_back(3);*/
		out_list_samples.clear();
		static std::vector<std::vector<int>> possible_hold_index_diff;
		if (possible_hold_index_diff.size()!=4)
			possible_hold_index_diff.resize(4);
		const int itr_index_diff_size=4;
		int itr_index_diff[itr_index_diff_size]={0,0,0,0};

		for (unsigned int i = 0; i < 4; i++)
		{
			std::vector<int> &possible_hold_diff_i=possible_hold_index_diff[i];
			possible_hold_diff_i.clear();

			addToSetHoldIDs(-1, possible_hold_diff_i);
			addToSetHoldIDs(from_hold_ids[i], possible_hold_diff_i);

			for (unsigned int j = 0; j < indices_lower_than[to_hold_id].size(); j++)
			{
				addToSetHoldIDs(indices_lower_than[to_hold_id][j], possible_hold_diff_i);
			}

			for (unsigned int j = 0; j < from_hold_ids.size(); j++)
			{
				if (mSampler::isInSetHoldIDs(from_hold_ids[j], indices_lower_than[to_hold_id]))
					addToSetHoldIDs(from_hold_ids[j], possible_hold_diff_i);
			}
		}
		bool flag_continue = true;
		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n(4,-1);
			//for (int i = 3; i >= 0; i--){sample_n.push_back(-1);}

			for (unsigned int i = 0; i < itr_index_diff_size; i++)
			{
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> &possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[i] = possible_hold_diff_i[itr_index_diff_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < itr_index_diff_size; i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= (int)possible_hold_index_diff[i].size())
				{
					if (i == itr_index_diff_size - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////

			/*if (mNode::isSetAEqualsSetB(dStance, sample_n) && mNode::isSetAEqualsSetB(fStance, from_hold_ids) && to_hold_id == 3)
			{
				int notifyme = 1;
			}*/

			if (sample_n[2] != to_hold_id && sample_n[3] != to_hold_id)
			{
				continue;
			}

			if (!isFromStanceToStanceValid(from_hold_ids, sample_n, false))
			{
				continue;
			}

			if (!isInSampledStanceSet(sample_n, out_list_samples))
				out_list_samples.push_back(sample_n);
		}
	}

	//this version not used anymore?
	std::vector<std::vector<int>> getListOfStanceSamplesAround(std::vector<int>& from_hold_ids, bool isInitialStance)
	{
		std::vector<std::vector<int>> list_samples;

		// this function assumes all from_hold_ids are greater than -1
		std::vector<std::vector<int>> workSpaceHoldIDs;
		getAllHoldsInRangeAroundAgent(from_hold_ids, workSpaceHoldIDs);

		std::vector<std::vector<int>> possible_hold_index_diff;
		std::vector<unsigned int> itr_index_diff;
		for (unsigned int i = 0; i < workSpaceHoldIDs.size(); i++)
		{
			std::vector<int> possible_hold_diff_i;

			std::vector<int> w_i = workSpaceHoldIDs[i];

			for (unsigned int j = 0; j < w_i.size(); j++)
			{
				addToSetHoldIDs(w_i[j], possible_hold_diff_i);
			}

			addToSetHoldIDs(from_hold_ids[i], possible_hold_diff_i);

			itr_index_diff.push_back(0);
			possible_hold_index_diff.push_back(possible_hold_diff_i);
		}

		bool flag_continue = true;
		while (flag_continue)
		{
			// create sample n
			std::vector<int> sample_n;
			for (int i = from_hold_ids.size() - 1; i >= 0; i--)
			{
				sample_n.push_back(-1);
			}
			for (unsigned int i = 0; i < itr_index_diff.size(); i++)
			{
				int itr_index_diff_i = itr_index_diff[i];
				std::vector<int> possible_hold_diff_i = possible_hold_index_diff[i];
				sample_n[i] = possible_hold_diff_i[itr_index_diff_i];
			}

			// increase itr num
			for (unsigned int i = 0; i < itr_index_diff.size(); i++)
			{
				itr_index_diff[i] = itr_index_diff[i] + 1;
				if (itr_index_diff[i] >= possible_hold_index_diff[i].size())
				{
					if (i == itr_index_diff.size() - 1)
					{
						flag_continue = false;
					}
					itr_index_diff[i] = 0;
				}
				else
				{
					break;
				}
			}

			///////////////////////////////////////////////////////////////
			// prior for adding sample_n to the list of possible samples //
			///////////////////////////////////////////////////////////////

			if (!isFromStanceToStanceValid(from_hold_ids, sample_n, isInitialStance))
			{
				continue;
			}

			list_samples.push_back(sample_n);
		}
		return list_samples;
	}

	mSampleStructure getSampleFrom(mNode* closestNode, std::vector<int>& chosen_desired_HoldsIds, bool flag_forShowing)
	{
//		printf("\n nh: 1,");
		if (closestNode == nullptr)
		{
			return mSampleStructure();
		}

//		printf(" 2,");
		if (flag_forShowing)
		{
			mSampleStructure ret_sample;
			ret_sample.closest_node_index = closestNode->nodeIndex;
			for (unsigned int i = 0; i < chosen_desired_HoldsIds.size(); i++)
			{
				if (chosen_desired_HoldsIds[i] >= 0)
					ret_sample.dPoint.push_back(getHoldPos(chosen_desired_HoldsIds[i]));
			}
			return ret_sample;
		}
//		printf(" 3,");
		std::vector<std::vector<size_t>> workSpaceHoldIDs;
		std::vector<Vector3> workspacePoints; // for debug visualization
		std::vector<Vector3> workSpacePointColor; // for debug visualization
		std::vector<bool> isIDInWorkSapceVector; // is current id in workspace or not

		getAllHoldsInRangeAroundAgent(closestNode, workSpaceHoldIDs, workspacePoints, workSpacePointColor, isIDInWorkSapceVector);
//		printf(" 4,");
//		std::vector<int> chosen_desired_HoldsIds = list_samples[ret_index];
		
		std::vector<int> rand_desired_HoldsIds;
		std::vector<int> sampled_rInitial_hold_ids;
		std::vector<Vector3> rndDesPos;
		Vector3 middle_point(0.0f, 0.0f, 0.0f);
		Vector3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		float num_initiated_holds = 0;

		std::vector<int> initial_holds_ids = closestNode->cNodeInfo.hold_bodies_ids;
		std::vector<Vector3> dVisualPoints; // for debug visualization
//		printf(" 5,");
		for (unsigned int i = 0; i < initial_holds_ids.size(); i++)
		{
			rand_desired_HoldsIds.push_back(chosen_desired_HoldsIds[i]);
			if (rand_desired_HoldsIds[i] == initial_holds_ids[i])
			{
				sampled_rInitial_hold_ids.push_back(initial_holds_ids[i]);
				if (initial_holds_ids[i] != -1)
				{
					rndDesPos.push_back(getHoldPos(initial_holds_ids[i]));
					middle_point += rndDesPos[i];
					if (max_point.z() < rndDesPos[i].z()){max_point.z() = rndDesPos[i].z();}
					num_initiated_holds++;
				}
				else
				{
					rndDesPos.push_back(Vector3(0.0f,0.0f,0.0f)); // some dummy var
				}
			}
			else
			{
				sampled_rInitial_hold_ids.push_back(-1);
				if (rand_desired_HoldsIds[i] != -1)
				{
					rndDesPos.push_back(getHoldPos(rand_desired_HoldsIds[i]));
					middle_point += rndDesPos[i];
					if (max_point.z() < rndDesPos[i].z()){max_point.z() = rndDesPos[i].z();}
					num_initiated_holds++;
				}
				else
				{
					rndDesPos.push_back(Vector3(0.0f,0.0f,0.0f)); // some dummy var
				}
			}
		}
//		printf(" 6,");
		middle_point /= (num_initiated_holds + 0.001f);

		Vector3 desired_trunk_pos(middle_point.x(), middle_point.y(), max_point.z() - boneLength / 4.0f);
//		dVisualPoints.push_back(desired_trunk_pos);
		rndDesPos.insert(rndDesPos.begin(), desired_trunk_pos);

		for (int i = rand_desired_HoldsIds.size() - 1; i >= 0; i--)
		{
			if (rand_desired_HoldsIds[i] != -1)
				dVisualPoints.push_back(getHoldPos(rand_desired_HoldsIds[i]));
		}
//		printf(" 7,");
		closestNode->addTriedHoldSet(rand_desired_HoldsIds);

		std::vector<robot_CPBP::ControlledPoses> sourceP;
		std::vector<Vector3> destinationP;
		setDesiredPosForOptimization(sourceP, destinationP, rndDesPos, rand_desired_HoldsIds, true);
//		printf(" 8,");
		return mSampleStructure(sourceP, destinationP, sampled_rInitial_hold_ids, rand_desired_HoldsIds, closestNode->nodeIndex, workspacePoints, workSpacePointColor, dVisualPoints);
	}

	mSampleStructure getSampleFrom(mNode* fromNode, mNode* toNode)
	{
		//printf("\n nn: 1,");
		//std::vector<std::vector<size_t>> workSpaceHoldIDs;
		//std::vector<Vector3> workspacePoints; // for debug visualization
		//std::vector<Vector3> workSpacePointColor; // for debug visualization
		//std::vector<bool> isIDInWorkSapceVector; // is current id in workspace or not

		//printf(" 2,");
		//getAllHoldsInRangeAroundAgent(fromNode, workSpaceHoldIDs, workspacePoints, workSpacePointColor, isIDInWorkSapceVector);

		//std::vector<int> rand_desired_HoldsIds = toNode->cNodeInfo.hold_bodies_ids;
		//std::vector<int> sampled_rInitial_hold_ids;

		//std::vector<int> initial_holds_ids = fromNode->cNodeInfo.hold_bodies_ids;
		//printf(" 3,");
		//for (unsigned int i = 0; i < initial_holds_ids.size(); i++)
		//{
		//	if (initial_holds_ids[i] == rand_desired_HoldsIds[i])
		//	{
		//		sampled_rInitial_hold_ids.push_back(initial_holds_ids[i]);
		//	}
		//	else
		//	{
		//		sampled_rInitial_hold_ids.push_back(-1);
		//	}
		//}
		//printf(" 4,");
		//std::vector<Vector3> dVisualPoints; // for debug visualization
		//for (int i = rand_desired_HoldsIds.size() - 1; i >= 0; i--)
		//{
		//	if (rand_desired_HoldsIds[i] != -1)
		//		dVisualPoints.push_back(getHoldPos(rand_desired_HoldsIds[i]));
		//}

		//printf(" 5,");
		//std::vector<robot_CPBP::ControlledPoses> sourceP;
		//std::vector<float> destinationA;
		//std::vector<Vector3> destinationP;

		//sourceP.push_back(robot_CPBP::ControlledPoses::Posture);
		//printf(" 6,");
		//for (unsigned int i = 0; i < toNode->cNodeInfo.toDesAngles.size(); i++)
		//{
		//	destinationA.push_back(toNode->cNodeInfo.toDesAngles[i]);
		//}
		//for (unsigned int i = 0; i < toNode->cNodeInfo.bodyStates.size(); i++)
		//{
		//	destinationP.push_back(toNode->cNodeInfo.bodyStates[i].getPos());
		//}
		//for (unsigned int i = 0; i < toNode->cNodeInfo.bodyStates.size(); i++)
		//{
		//	destinationP.push_back(toNode->cNodeInfo.bodyStates[i].getPos());
		//}
		//printf(" 7,");
		//return mSampleStructure(sourceP, destinationA, destinationP, sampled_rInitial_hold_ids, rand_desired_HoldsIds
		//				, fromNode->nodeIndex, workspacePoints, workSpacePointColor, dVisualPoints);

		 
		mSampleStructure nSample = getSampleFrom(fromNode, toNode->cNodeInfo.hold_bodies_ids, false);

		for (unsigned int i = 0; i < toNode->cNodeInfo.toDesAngles.size(); i++)
		{
			nSample.destinationA.push_back(toNode->cNodeInfo.toDesAngles[i]);
		}

		return nSample;
	}

	static float getRandomBetween_01()
	{
		return ((float)rand()) / (float)RAND_MAX;
	}
	
	static int getRandomIndex(unsigned int iArraySize)
	{
		if (iArraySize == 0)
			return -1;
		int m_index = rand() % iArraySize;

		return m_index;
	}

	static bool isInSetHoldIDs(int hold_id, std::vector<int>& iSetIDs)
	{
		for (unsigned int i = 0; i < iSetIDs.size(); i++)
		{
			if (iSetIDs[i] == hold_id)
			{
				return true;
			}
		}
		return false;
	}
	
	static bool addToSetHoldIDs(int hold_id, std::vector<int>& iSetIDs)
	{
		if (!isInSetHoldIDs(hold_id, iSetIDs))
		{
			iSetIDs.push_back(hold_id);
			return true;
		}
		return false;
	}

	static void removeFromSetHoldIDs(int hold_id, std::vector<int>& iSetIDs)
	{
		for (unsigned int i = 0; i < iSetIDs.size(); i++)
		{
			if (iSetIDs[i] == hold_id)
			{
				iSetIDs.erase(iSetIDs.begin() + i);
				return;
			}
		}
	}

	static bool isInSampledStanceSet(std::vector<int>& sample_i, std::vector<std::vector<int>>& nStances)
	{
		for (unsigned int i = 0; i < nStances.size(); i++)
		{
			std::vector<int> t_i = nStances[i];
			bool flag_found_try = mNode::isSetAEqualsSetB(t_i, sample_i);
			if (flag_found_try)
			{
				return true;
			}
		}
		return false;
	}

private:

	void fillInLowerHigherHoldIndices()
	{
		float max_radius_around_hand = 1.0f * climberRadius;

		for (unsigned int  k = 0; k < myHoldPoints.size(); k++)
		{
			Vector3 dHoldPos = myHoldPoints[k];
			std::vector<size_t> ret_holds_ids = getPointsInRadius(myHoldPoints[k], max_radius_around_hand);
			std::vector<int> lower_holds_ids;
			std::vector<int> higher_holds_ids;
			for (unsigned int i = 0; i < ret_holds_ids.size(); i++)
			{
				bool flag_add = true;
				Vector3 hold_pos_i = getHoldPos(ret_holds_ids[i]);

				float cDis = (hold_pos_i - dHoldPos).norm();

				if (cDis < 0.01f)
				{
					lower_holds_ids.push_back(ret_holds_ids[i]);
					higher_holds_ids.push_back(ret_holds_ids[i]);
					continue;
				}

				float angle_btw_l = SimulationContext::getAngleBtwVectorsXZ(Vector3(1.0f, 0.0f, 0.0f), hold_pos_i - dHoldPos);
				if (((angle_btw_l <= 0) && (angle_btw_l >= -PI)) || (angle_btw_l >= PI)) 
				{
					lower_holds_ids.push_back(ret_holds_ids[i]);
				}
				if (((angle_btw_l >= 0 ) && (angle_btw_l <= PI)) || (angle_btw_l <= -PI)) 
				{
					higher_holds_ids.push_back(ret_holds_ids[i]);
				}
			}
			indices_lower_than.push_back(lower_holds_ids);
			indices_higher_than.push_back(higher_holds_ids);
		}
		return;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int getClosest_WSHoldIndex_ToDesiredPos(std::vector<size_t>& workSpaceSet, int desiredID, int currentID, Vector3& cPos)
	{
		int ret_index = currentID;
		if (desiredID == -1)
			return ret_index;

		Vector3 dPos = getHoldPos(desiredID);
		float cDis = (dPos - cPos).norm();
		ret_index = currentID;
		for (unsigned int i = 0; i < workSpaceSet.size(); i++)
		{
			Vector3 nPosHold = getHoldPos(workSpaceSet[i]);
			float nDis = (dPos - nPosHold).norm();
			if (nDis < cDis)
			{
				ret_index = workSpaceSet[i];
				cDis = nDis;
			}
		}
		return ret_index;
	}

	bool earlyAcceptOfSample(std::vector<int>& sample_desired_hold_ids, bool isStanceGiven)
	{
		if (!isStanceGiven)
		{
			if (sample_desired_hold_ids[0] == sample_desired_hold_ids[1] && 
				(sample_desired_hold_ids[2] == sample_desired_hold_ids[0] || sample_desired_hold_ids[3] == sample_desired_hold_ids[0])) // if hands and legs are on the same hold
				return false;
			if (sample_desired_hold_ids[2] == sample_desired_hold_ids[3] && 
				(sample_desired_hold_ids[2] == sample_desired_hold_ids[0] || sample_desired_hold_ids[2] == sample_desired_hold_ids[1])) // if hands and legs are on the same hold
				return false;

			if (mNode::isSetAGreaterThan(sample_desired_hold_ids, -1))
			{
				std::vector<int> diff_hold_ids;
				for (unsigned int i = 0; i < sample_desired_hold_ids.size(); i++)
				{
					if (sample_desired_hold_ids[i] != -1)
						mSampler::addToSetHoldIDs(sample_desired_hold_ids[i], diff_hold_ids);
				}
				if (diff_hold_ids.size() <= 2)
				{
					return false;
				}
			}
		}
		std::vector<Vector3> sample_desired_hold_points;
		float mSize = 0;
		Vector3 midPoint = getHoldStancePosFrom(sample_desired_hold_ids, sample_desired_hold_points, mSize);

		if (mSize == 0)
		{
			return false;
		}

		// early reject of the sample (do not try the sample, because it is not reasonable)

		float coefficient_hand = 1.5f;
		float coefficient_leg = 1.2f;
		float coefficient_all = 1.1f;
		// if hands or legs distance are violating
		if (sample_desired_hold_ids[0] != -1 && sample_desired_hold_ids[1] != -1)
		{
			float dis_ll = (sample_desired_hold_points[0] - sample_desired_hold_points[1]).norm();
			if (dis_ll > coefficient_leg * climberLegLegDis)
			{
				return false;
			}
		}
		
		if (sample_desired_hold_ids[2] != -1 && sample_desired_hold_ids[3] != -1)
		{
			float dis_hh = (sample_desired_hold_points[2] - sample_desired_hold_points[3]).norm();
			if (dis_hh > coefficient_hand * climberHandHandDis)
			{
				return false;
			}
		}

		if (sample_desired_hold_ids[0] != -1 && sample_desired_hold_ids[2] != -1)
		{
			float dis_h1l1 = (sample_desired_hold_points[2] - sample_desired_hold_points[0]).norm();

			if (dis_h1l1 > coefficient_all * climberRadius)
				return false;
		}
		
		if (sample_desired_hold_ids[0] != -1 && sample_desired_hold_ids[3] != -1)
		{
			float dis_h2l1 = (sample_desired_hold_points[3] - sample_desired_hold_points[0]).norm();
			if (dis_h2l1 > coefficient_all * climberRadius)
				return false;
		}
		
		if (sample_desired_hold_ids[1] != -1 && sample_desired_hold_ids[2] != -1)
		{
			float dis_h1l2 = (sample_desired_hold_points[2] - sample_desired_hold_points[1]).norm();
			if (dis_h1l2 > coefficient_all * climberRadius)
				return false;
		}

		if (sample_desired_hold_ids[1] != -1 && sample_desired_hold_ids[3] != -1)
		{
			float dis_h2l2 = (sample_desired_hold_points[3] - sample_desired_hold_points[1]).norm();
			if (dis_h2l2 > coefficient_all * climberRadius)
				return false;
		}
		
		return true;
	}

	bool isFromStanceCloseEnough(std::vector<int>& initial_holds_ids, std::vector<int>& iSample_desired_hold_ids)
	{
		int m_count = 0; 
		for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
		{
			if (iSample_desired_hold_ids[i] == initial_holds_ids[i])
			{
				m_count++;
			}
		}

		if (m_count >= 2)
		{
			return true;
		}
		return false;
	}

	bool acceptEqualPart(std::vector<int>& sample_n, std::vector<int>&  initial_holds_ids, std::vector<int>& iSample_desired_hold_ids)
	{
		for (unsigned int i = 0; i < iSample_desired_hold_ids.size(); i++)
		{
			if (iSample_desired_hold_ids[i] == sample_n[i] && initial_holds_ids[i] != sample_n[i])
			{
				return true;
			}

		}
		return false;
	}

	bool acceptDirectionLegsAndHands(Vector3 mPoint, std::vector<int>& sample_n_ids, std::vector<Vector3>& sample_n_points)
	{
		/*for (unsigned int i = 0; i < sample_n_ids.size(); i++)
		{
			if (sample_n_ids[i] > 0)
			{
				float angle_btw = SimulationContext::getAngleBtwVectorsXZ(Vector3(1.0f, 0.0f, 0.0f), sample_n_points[i] - mPoint);
				if (i == 0 || i == 1)
				{
					if (!(angle_btw < 0.3 * PI && angle_btw > -PI || angle_btw > 0.7f * PI))
					{
						return false;
					}
				}
				else
				{
					if (!(angle_btw > -0.3 * PI && angle_btw < PI || angle_btw < -0.7f * PI))
					{
						return false;
					}
				}
			}
		}*/
		if (sample_n_ids[0] != -1 || sample_n_ids[1] != -1)
		{
			float z = 0;
			if (sample_n_ids[2] != -1)
			{
				z = std::max<float>(z, sample_n_points[2].z());
			}
			if (sample_n_ids[3] != -1)
			{
				z = std::max<float>(z, sample_n_points[3].z());
			}

			if (sample_n_ids[0] != -1)
			{
				if (z < sample_n_points[0].z())
				{
					return false;
				}
			}
			if (sample_n_ids[1] != -1)
			{
				if (z < sample_n_points[1].z())
				{
					return false;
				}
			}
		}

		return true;
	}

	//////////////////////////////////////////////////// general use for sampler ////////////////////////////////////////////////////

	bool isAllowedHandsLegsInDSample(std::vector<int>& initial_hold_ids, std::vector<int>& sampled_desired_hold_ids, bool isStanceGiven)
	{
		static std::vector<int> sample_initial_id;  //static to prevent heap alloc (a quick hack)
		sample_initial_id.clear();
		if (isStanceGiven)
		{
			if (initial_hold_ids[2] == -1 && initial_hold_ids[3] == -1)
			{
				if (sampled_desired_hold_ids[2] != -1 || sampled_desired_hold_ids[3] != -1)
				{
					return true;
				}
			}
			else if ((initial_hold_ids[2] != -1 || initial_hold_ids[3] != -1) && initial_hold_ids[0] == -1 && initial_hold_ids[1] == -1)
			{
				if (initial_hold_ids[2] == -1 && sampled_desired_hold_ids[2] != -1)
				{
					return true;
				}
				if (initial_hold_ids[3] == -1 && sampled_desired_hold_ids[3] != -1)
				{
					return true;
				}
				if (sampled_desired_hold_ids[0] != -1 || sampled_desired_hold_ids[1] != -1)
				{
					return true;
				}
			}
		}

		for (unsigned int i = 0; i < initial_hold_ids.size(); i++)
		{
			if (sampled_desired_hold_ids[i] == initial_hold_ids[i])
			{
				sample_initial_id.push_back(initial_hold_ids[i]);
			}
			else
			{
				sample_initial_id.push_back(-1);
			}
		}

		for (unsigned int i = 0; i < sample_initial_id.size(); i++)
		{
			if (sample_initial_id[i] == -1)
			{
				if (!isAllowedToReleaseHand_i(sample_initial_id, i) || !isAllowedToReleaseLeg_i(sample_initial_id, i))
					return false;
			}
		}
		return true;
	}

	bool isAllowedToReleaseHand_i(std::vector<int>& sampled_rInitial_hold_ids, int i)
	{
		if (i == sampled_rInitial_hold_ids.size() - 1 && sampled_rInitial_hold_ids[i - 1] == -1)
		{
			return false;
		}

		if (i == sampled_rInitial_hold_ids.size() - 2 && sampled_rInitial_hold_ids[i + 1] == -1)
		{
			return false;
		}

		if (sampled_rInitial_hold_ids[0] == -1 && sampled_rInitial_hold_ids[1] == -1 && i >= 2) // check feet
		{
			return false;
		}
		return true;
	}

	bool isAllowedToReleaseLeg_i(std::vector<int>& sampled_rInitial_hold_ids, int i)
	{
		if (sampled_rInitial_hold_ids[sampled_rInitial_hold_ids.size() - 1] != -1 && sampled_rInitial_hold_ids[sampled_rInitial_hold_ids.size() - 2] != -1)
		{
			return true;
		}

		if (i == 1 && sampled_rInitial_hold_ids[i - 1] == -1)
		{
			return false;
		}

		if (i == 0 && sampled_rInitial_hold_ids[i + 1] == -1)
		{
			return false;
		}

		return true;
	}

	void getAllHoldsInRangeAroundAgent(std::vector<int>& from_hold_ids, std::vector<std::vector<int>>& workSpaceHoldIDs)
	{

		std::vector<int> sampled_rInitial_hold_ids = from_hold_ids;

		for (unsigned int i = 0; i < sampled_rInitial_hold_ids.size(); i++)
		{
			std::vector<int> workSpaceHolds = getWorkSpaceAround(robot_CPBP::ControlledPoses(robot_CPBP::ControlledPoses::LeftLeg + i), from_hold_ids);
			workSpaceHoldIDs.push_back(workSpaceHolds);
		}

	}

	void getAllHoldsInRangeAroundAgent(mNode* iNode, std::vector<std::vector<size_t>>& workSpaceHoldIDs
		, std::vector<Vector3>& workspacePoints, std::vector<Vector3>& workSpacePointColor, std::vector<bool>& isIDInWorkSapceVector)
	{

		std::vector<int> sampled_rInitial_hold_ids = iNode->cNodeInfo.hold_bodies_ids;

		std::vector<Vector3> colors;
		colors.push_back(Vector3(255, 0, 0) / 255.0f); // blue cv::Scalar(255, 0, 0)
		colors.push_back(Vector3(0, 255, 0) / 255.0f); // green cv::Scalar(0, 255, 0)
		colors.push_back(Vector3(0, 0, 255) / 255.0f); // red cv::Scalar(0, 0, 255)
		colors.push_back(Vector3(255, 255, 255) / 255.0f); // whilte cv::Scalar(255, 255, 255)

		for (unsigned int i = 0; i < sampled_rInitial_hold_ids.size(); i++)
		{
			bool isIDInWorkSapce = false;
			std::vector<size_t> workSpaceHolds = getWorkSpaceAround(robot_CPBP::ControlledPoses(robot_CPBP::ControlledPoses::LeftLeg + i), iNode, isIDInWorkSapce);
			isIDInWorkSapceVector.push_back(isIDInWorkSapce);
			workSpaceHoldIDs.push_back(workSpaceHolds);
			
			for (unsigned int j = 0; j < workSpaceHolds.size(); j++)
			{
				Vector3 hold_pos_j = getHoldPos(workSpaceHolds[j]);
				workspacePoints.push_back(hold_pos_j);
				workSpacePointColor.push_back(colors[i]);
			}
		}

	}

	std::vector<int> getWorkSpaceAround(robot_CPBP::ControlledPoses iPosSource, std::vector<int>& from_hold_ids)
	{
		float max_radius_search = climberRadius / 1.3f; //2.6f * boneLength;
		float min_radius_search = climberRadius * 0.01f; //0.4f * boneLength;

		// evaluating starting pos
		std::vector<Vector3> from_hold_points; float size_d = 0;
		Vector3 trunk_pos = getHoldStancePosFrom(from_hold_ids, from_hold_points, size_d);
		trunk_pos[1] = 0;

		std::vector<size_t> holds_ids = getPointsInRadius(trunk_pos, max_radius_search);

		std::vector<int> desired_holds_ids;
		for (unsigned int i = 0; i < holds_ids.size(); i++)
		{
			Vector3 Pos_i = getHoldPos(holds_ids[i]);

			bool flag_add_direction_hold = false;

			float m_angle = SimulationContext::getAngleBtwVectorsXZ(Vector3(1.0f, 0.0f, 0.0f), Pos_i - trunk_pos);


			if ((m_angle <= 0.3f * PI) && (m_angle >= -PI) || (m_angle >= 0.3f * PI)) 
			{
				if (iPosSource == robot_CPBP::ControlledPoses::LeftLeg || iPosSource == robot_CPBP::ControlledPoses::RightLeg)
				{
					flag_add_direction_hold = true;
				}
			}
			if ((m_angle >= -0.3f * PI) && (m_angle <= PI) || (m_angle >= -0.3f * PI)) 
			{
				if (iPosSource == robot_CPBP::ControlledPoses::LeftHand || iPosSource == robot_CPBP::ControlledPoses::RightHand)
				{
					flag_add_direction_hold = true;
				}
			}

			if (flag_add_direction_hold)
			{
				float dis = (Pos_i - trunk_pos).norm();
				if (dis < max_radius_search && dis > min_radius_search)
				{
					desired_holds_ids.push_back(holds_ids[i]);
				}
			}
		}

		return desired_holds_ids;
	}

	std::vector<size_t> getWorkSpaceAround(robot_CPBP::ControlledPoses iPosSource, mNode* iNode, bool &isIDInWorkSapce)
	{
		float max_radius_search = climberRadius / 1.9f; //2.6f * boneLength;
		float min_radius_search = climberRadius * 0.08f; //0.4f * boneLength;

		isIDInWorkSapce = false;
		int cID = iNode->cNodeInfo.hold_bodies_ids[iPosSource - robot_CPBP::ControlledPoses::LeftLeg];

		// evaluating starting pos
		Vector3 sPos = iNode->cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		sPos[1] = 0;

		std::vector<size_t> holds_ids = getPointsInRadius(sPos, max_radius_search);

		Vector3 trunk_pos = iNode->cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		trunk_pos[1] = 0;
		Vector3 trunkDir = iNode->cNodeInfo.getBodyDirectionZ(SimulationContext::BodyName::BodyTrunk);//(SimulationContext::BodyName::BodyTrunkUpper) - trunk_pos;
		trunkDir[1] = 0;
		trunkDir = trunkDir.normalized();

		std::vector<size_t> desired_holds_ids;
		for (unsigned int i = 0; i < holds_ids.size(); i++)
		{
			Vector3 Pos_i = getHoldPos(holds_ids[i]);

			bool flag_add_direction_hold = false;
			float angle_btw_trunk = SimulationContext::getAngleBtwVectorsXZ(Pos_i - sPos, trunkDir); // we have a limit for turning direction with relation to trunk posture
			
			float angle_btw = angle_btw_trunk;

			float angle_btw_l = SimulationContext::getAngleBtwVectorsXZ(Vector3(1.0f, 0.0f, 0.0f), Pos_i - trunk_pos);
			switch (iPosSource)
			{
			case robot_CPBP::LeftLeg:
				if (angle_btw >= -PI && angle_btw <= -PI * 0.05f)
					flag_add_direction_hold = true;

				if (angle_btw <= PI && angle_btw > PI * 0.75f)
					flag_add_direction_hold = true;

				if (!(angle_btw_l < 0 && angle_btw_l > -PI))
					flag_add_direction_hold = false;
				break;
			case robot_CPBP::RightLeg:
				if (angle_btw >= PI * 0.05f && angle_btw <= PI)
					flag_add_direction_hold = true;

				if (angle_btw >= -PI && angle_btw <= -PI * 0.75f)
					flag_add_direction_hold = true;

				if (!(angle_btw_l < 0 && angle_btw_l > -PI))
					flag_add_direction_hold = false;
				break;
			case robot_CPBP::LeftHand:
				if (angle_btw >= -PI * 0.75f && angle_btw <= 0)
					flag_add_direction_hold = true;
				if (angle_btw >= 0 && angle_btw <= PI * 0.25f)
					flag_add_direction_hold = true;
				break;
			case robot_CPBP::RightHand:
				if (angle_btw >= 0 && angle_btw <= PI * 0.75f)
					flag_add_direction_hold = true;
				if (angle_btw >= -PI * 0.25f && angle_btw <= 0)
					flag_add_direction_hold = true;
				break;
			default:
				break;
			}

			if (flag_add_direction_hold)
			{
				float dis = (Pos_i - sPos).norm();
				if (dis < max_radius_search && dis > min_radius_search)
				{
					if (cID == holds_ids[i])
					{
						isIDInWorkSapce = true;
					}
					desired_holds_ids.push_back(holds_ids[i]);
				}
			}
		}

		return desired_holds_ids;
	}

	void setDesiredPosForOptimization(std::vector<robot_CPBP::ControlledPoses>& sourceP, std::vector<Vector3>& destinationP
		, std::vector<Vector3>& iRandDesPos, std::vector<int>& rand_desired_HoldsIds, bool flag_set_trunk)
	{
		sourceP.push_back(robot_CPBP::ControlledPoses::mHead);
		if (flag_set_trunk)
			sourceP.push_back(robot_CPBP::ControlledPoses::MiddleTrunk);
		Vector3 headDir(0.0f, 0.0f, 1.0f);
		destinationP.push_back(headDir); // head's angle pos and velocity for head
		if (flag_set_trunk)
			destinationP.push_back(iRandDesPos[0]); // middle trunk's pos

		for (unsigned int i = 0; i < rand_desired_HoldsIds.size(); i++)
		{
			if (rand_desired_HoldsIds[i] != -1)
			{
				sourceP.push_back((robot_CPBP::ControlledPoses)(robot_CPBP::ControlledPoses::LeftLeg + i));
				if (flag_set_trunk)
					destinationP.push_back(iRandDesPos[i + 1]); // pos (first one is for trunk pos)
				else
					destinationP.push_back(iRandDesPos[i]); // pos 
			}
		}

		return;
	}

	/////////////////////////////////////////// handling holds using kd-tree ///////////////////////////////////////////////////////
	
	std::vector<double> getHoldKey(Vector3& c)
	{
		std::vector<double> rKey;

		rKey.push_back(c.x());
		rKey.push_back(c.y());
		rKey.push_back(c.z());

		return rKey;
	}
public:

	/////////////////////////////////////////// handling holds using kd-tree ///////////////////////////////////////////////////////

	int getClosestPoint(Vector3& qP)
	{
		std::vector<int> ret_index = mHoldsKDTree.nearest(getHoldKey(qP), 1);

		if (ret_index.size() > 0)
			return ret_index[0];
		return -1;
	}

	std::vector<size_t> getPointsInRadius(Vector3& qP, float r)
	{
		std::vector<size_t> ret_index;
		for (unsigned int i = 0; i < myHoldPoints.size(); i++)
		{
			Vector3 hold_i = myHoldPoints[i];
			float cDis = (hold_i - qP).norm();
			if (cDis < r)
			{
				ret_index.push_back(i);
			}
		}
		return ret_index;
	}

};

class mTestControllerClass
{
public:
	mSampler mClimberSampler;
	robot_CPBP* mController;
	SimulationContext* mContextRRT;
	mSampleStructure mSample;

	std::vector<int> desired_holds_ids;

	// handling falling problem of the climber
	BipedState lOptimizationBipedState;
	bool isCopiedForDesiredHoldIDs;

	mTestControllerClass(SimulationContext* iContextRRT, robot_CPBP* iController)
		:mClimberSampler(iContextRRT)
	{
		mController = iController;
		mContextRRT = iContextRRT;

		for (unsigned int i = 0; i < iController->startState.hold_bodies_ids.size(); i++)
		{
			mSample.initial_hold_ids.push_back(iController->startState.hold_bodies_ids[i]);
			mSample.desired_hold_ids.push_back(iController->startState.hold_bodies_ids[i]);
			desired_holds_ids.push_back(-1);
		}

		lOptimizationBipedState = mController->startState.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID());
		isCopiedForDesiredHoldIDs = true;

		desired_holds_ids[2] = -1;
		desired_holds_ids[3] = 3;
	}

	void runTest(bool advance_time, bool& revertToLastStableState)
	{
		//getMouseInfo();
		if (revertToLastStableState)
		{
			revertToLastStableState = !revertToLastStableState;
			mController->startState = lOptimizationBipedState;

			if (!mNode::isSetAEqualsSetB(mController->startState.hold_bodies_ids, mSample.desired_hold_ids))
			{
				setArmsLegsHoldPoses();
			}
			mController->takeStep();
		}
		else
		{
			updateSampleInfo();

			mSteerFunc(advance_time);
			// connect contact pos i to the desired hold pos i if some condition (for now just distance) is met
			m_Connect_Disconnect_ContactPoint(mSample.desired_hold_ids);

			mSample.draw_ws_points(Vector3(0.0f,0.0f,0.0f));

			if (mNode::isSetAEqualsSetB(mController->startState.hold_bodies_ids, mSample.desired_hold_ids) && !isCopiedForDesiredHoldIDs)
			{
				// save last optimization state
				lOptimizationBipedState = mController->startState.getNewCopy(lOptimizationBipedState.saving_slot_state, mContextRRT->getMasterContextID());
				isCopiedForDesiredHoldIDs = true;
			}
		}
	}

private:
	//void getMouseInfo()
	//{
	//	int xPos, yPos, mButton;
	//	dsMyGetMouseInfo(xPos, yPos, mButton);
	//	float xMouse = (2.0f * xPos) / 800 - 1.0f;
	//	float yMouse = 1 - (2.0f * yPos) / 600;
	////	dInvertPDMatrix
	//	printf("X:%f, Y:%f, B:%d \n",xMouse, yMouse, mButton);
	//}

	void updateSampleInfo()
	{
		if (!mNode::isSetAEqualsSetB(desired_holds_ids, mSample.desired_hold_ids))
		{
			mSample.destinationP.clear();
			mSample.sourceP.clear();
			mSample.dPoint.clear();

			float avgX = 0;
			float maxZ = -FLT_MAX;
			int numDesiredHolds = 0;
			for (unsigned int i = 0; i < desired_holds_ids.size(); i++)
			{
				if (desired_holds_ids[i] != mSample.desired_hold_ids[i])
				{
					mSample.desired_hold_ids[i] = desired_holds_ids[i];
					mSample.initial_hold_ids[i] = -1;
				}
				else
				{
					mSample.initial_hold_ids[i] = mSample.desired_hold_ids[i];
				}
				if (desired_holds_ids[i] != -1)
				{
					Vector3 dPos = mClimberSampler.getHoldPos(desired_holds_ids[i]);

					mSample.sourceP.push_back((robot_CPBP::ControlledPoses)(robot_CPBP::ControlledPoses::LeftLeg + i));
					mSample.destinationP.push_back(dPos);
					mSample.dPoint.push_back(dPos);

					avgX += dPos.x();
					if (dPos.z() > maxZ)
					{
						maxZ = dPos.z();
					}
					numDesiredHolds++;
				}
			}
			
			if (maxZ >= 0)
			{
				avgX /= (float)(numDesiredHolds + 0.0f);

				Vector3 tPos(avgX, 0, maxZ - boneLength / 4.0f);

				mSample.sourceP.push_back(robot_CPBP::ControlledPoses::MiddleTrunk);
				mSample.destinationP.push_back(tPos);
				mSample.dPoint.push_back(tPos);
			}

			setArmsLegsHoldPoses();
			isCopiedForDesiredHoldIDs = false;
		}
	}

	//Not in use anymore! See the other mSteerFunc
	void mSteerFunc(bool advance_time)
	{
		mController->loadPhysicsToMaster(false);

		mController->optimize_the_cost(advance_time, mSample.sourceP, mSample.destinationP, mSample.destinationA);

		// check changing of the cost
		float costImprovement= mSample.cOptimizationCost-mController->current_cost;

		rcPrintString("Traj. cost improvement %f",costImprovement);
		if (costImprovement < noCostImprovementThreshold)
		{
			mSample.numItrFixedCost++;
		}
		else
		{
			mSample.numItrFixedCost = 0;
			mSample.cOptimizationCost = mController->current_cost;
		}

		//apply the best control to get the start state of next frame
		if (advance_time)
		{
			bool flagAddSimulation = mController->advance_simulation();
			if (flagAddSimulation)
			{
//				BipedState nState = mController->startState.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID());
//				mSample.statesFromTo.push_back(nState);
			}
			if (!flagAddSimulation)
			{
				mSample.isOdeConstraintsViolated = true;
			}
		}
		return;
	}

	void setArmsLegsHoldPoses()
	{
		mController->startState.hold_bodies_ids = mSample.initial_hold_ids;
		for (unsigned int i = 0; i < mController->startState.hold_bodies_ids.size(); i++)
		{
			if (mController->startState.hold_bodies_ids[i] != -1)
			{
				Vector3 hPos_i = mClimberSampler.getHoldPos(mController->startState.hold_bodies_ids[i]);
				Vector3 endPos_i = mController->startState.getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg + i);
				float cDis = (hPos_i - endPos_i).norm();
				if (cDis > 0.5f * holdSize)
				{
					mController->startState.hold_bodies_ids[i] = -1;
				}
			}
		}
		return;
	}

	void m_Connect_Disconnect_ContactPoint(std::vector<int>& desired_holds_ids)
	{
		float min_reject_angle = (PI / 2) - (0.3f * PI);
		//float max_acceptable_angle = 1.3f * PI;

		for (unsigned int i = 0; i < desired_holds_ids.size(); i++)
		{
			if (desired_holds_ids[i] != -1)
			{
				Vector3 hold_pos_i = mClimberSampler.getHoldPos(desired_holds_ids[i]);
				Vector3 contact_pos_i = mController->startState.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + i);

				float dis_i = (hold_pos_i - contact_pos_i).norm();

				if (dis_i < 0.25f * holdSize)
				{
					if (i <= 1) // left leg and right leg
					{
						Vector3 dir_contact_pos = -mController->startState.getBodyDirectionZ(SimulationContext::ContactPoints::LeftLeg + i);
						//mController->startState.bodyStates[SimulationContext::ContactPoints::LeftLeg + i].pos;
						float m_angle_btw = SimulationContext::getAbsAngleBtwVectors(-Vector3(0.0f, 0.0f, 1.0f), dir_contact_pos);

						if (m_angle_btw > min_reject_angle)
						{
							mController->startState.hold_bodies_ids[i] = desired_holds_ids[i];
						}
						else
						{
							mController->startState.hold_bodies_ids[i] = -1;
						}
					}
					else
					{
						mController->startState.hold_bodies_ids[i] = desired_holds_ids[i];
					}
				}
				else if (dis_i > 0.5f * holdSize)
				{
					mController->startState.hold_bodies_ids[i] = -1;
				}
			}
			else
			{
				mController->startState.hold_bodies_ids[i] = -1;
			}
		}
	}

}* mTestClimber;

class mStanceNode
{
public:
	bool isItExpanded; // for building the graph
	std::vector<int> hold_ids; // ll, rl, lh, rh

	int stanceIndex;
	std::vector<int> parentStanceIds; // parent Nodes
	std::vector<bool> isItTried_to_father;

	std::vector<int> childStanceIds; // childNodes

	float nodeCost; // cost of standing at node of graph
	std::vector<float> cost_transition_to_child;
	std::vector<float> cost_moveLimb_to_child;
	std::vector<bool> isItTried_to_child;

	float g_AStar;
	float h_AStar;

	bool dijkstraVisited;

//	int count_u;
	int childIndexInbFather;
	int bFatherStanceIndex;
	int bChildStanceIndex;

	mStanceNode(std::vector<int>& iInitStance)
	{
		g_AStar = FLT_MAX;
		h_AStar = FLT_MAX;
		bFatherStanceIndex = -1;
		bChildStanceIndex = -1;
		childIndexInbFather = -1;
//		count_u = 0;

		hold_ids = iInitStance;
		isItExpanded = false;
		stanceIndex = -1;
	}
};

class mStancePathNode // for A* prune
{
public:
	mStancePathNode(std::vector<int>& iPath, float iG, float iH)
	{
		_path.reserve(100);

		mG = iG;
		mH = iH;
		for (unsigned int i = 0; i < iPath.size(); i++)
		{
			_path.push_back(iPath[i]);
		}
	}

	std::vector<int> addToEndPath(int stanceID)
	{
		std::vector<int> nPath;
		for (unsigned int i = 0; i < _path.size(); i++)
		{
			nPath.push_back(_path[i]);
		}
		nPath.push_back(stanceID);
		return nPath;
	}

	std::vector<int> _path;
	float mG;
	float mH;
};


static inline uint32_t stanceToKey(const std::vector<int>& stance)
{
	uint32_t result=0;
	for (int i=0; i<4; i++)
	{
		uint32_t uNode=(uint32_t)(stance[i]+1);  //+1 to handle the -1
		/*if (!(uNode>=0 && uNode<256))
			Debug::throwError("Invalid hold index %d!\n",uNode);*/
		result=(result+uNode);
		if (i<3)
			result=result << 8; //shift, assuming max 256 holds
	}
	return result;
}

class mStanceGraph
{
	std::vector<int> initialStance;
	float _MaxTriedTransitionCost;
public:
	enum mAlgSolveGraph{myAlgAStar = 0, AStarPrune = 1, myAStarPrune = 2, myDijkstraHueristic = 3};

	std::vector<std::vector<int>> m_found_paths;
	std::unordered_map<uint32_t, int> stanceToNode; //key is uint64_t computed from a stance (i.e., a vector of 4 ints), data is index to the stance nodes vector
	DynamicMinMax minmax;

	int timeOfCreation; // in mili seconds
	std::vector<int> timeOfFindingPaths; //in mili seconds

	void reset()
	{
		retPath.clear();
		m_found_paths.clear();
		stanceToNode.clear();
		timeOfFindingPaths.clear();

		goal_stances.clear();
		stance_nodes.clear();

		maxGraphDegree = 0;
		stanceToNode.rehash(1000003);  //a prime number of buckets to minimize different keys mapping to same bucket
		mClimberSampler = nullptr;

		_MaxTriedTransitionCost = 100000.0f;
		//preVal = -FLT_MAX; // for debugging
		initializeOpenListAStarPrune();
	}

	mStanceGraph()
	{
		maxGraphDegree = 0;
		stanceToNode.rehash(1000003);  //a prime number of buckets to minimize different keys mapping to same bucket
		mClimberSampler = nullptr;

		_MaxTriedTransitionCost = 100000.0f;
		//preVal = -FLT_MAX; // for debugging
		initializeOpenListAStarPrune();
	}

	~mStanceGraph()
	{
		char buff[100];
		sprintf_s(buff, "TestFilesClimber\\FileStanceGraphCol%dRow%dT%d.txt", nColumns, nRows, nTestNum);
		
		FILE* rwFileStream;
		std::string mfilename = buff;
		
		fopen_s(&rwFileStream , mfilename.c_str(), "w");

		if (rwFileStream)
		{
			fprintf(rwFileStream, "%d \n", stance_nodes.size());
			fprintf(rwFileStream, "%d \n", timeOfCreation);

			for (unsigned int i = 0; i < timeOfFindingPaths.size(); i++)
			{
				if (i < timeOfFindingPaths.size() - 1)
					fprintf(rwFileStream, "%d,", timeOfFindingPaths[i]);
				else
					fprintf(rwFileStream, "%d", timeOfFindingPaths[i]);
			}
		
			fclose(rwFileStream);
		}
	}

	void initializeOpenListAStarPrune()
	{
		openListPath.clear();
		std::vector<int> rootPath; rootPath.push_back(0);
		openListPath.push_back(mStancePathNode(rootPath,0,0));
	}

	float getGUpdatedCostNode(int _tNode)
	{
		bool flag_continue = true;
		int cNode = _tNode;
		float tCost = 0.0f;
		while (flag_continue)
		{
			int bFather = stance_nodes[cNode].bFatherStanceIndex;
			if (bFather != -1)
			{
				mStanceNode& fNode = stance_nodes[bFather];
				int child_index = getChildIndex(bFather,cNode);
				float cCost = fNode.cost_transition_to_child[child_index] 
							+ fNode.cost_moveLimb_to_child[child_index] 
							+ stance_nodes[child_index].nodeCost;
				tCost += cCost;
				cNode = bFather;
			}
			else
			{
				flag_continue = false;
			}
		}
		return tCost;
	}

	float getHeuristicCostNodeToNode(int _fNode, int _tNode)
	{
		mStanceNode& fNode = stance_nodes[_fNode];
		mStanceNode& tNode = stance_nodes[_tNode];
		int c = getChildIndex(_fNode, _tNode);
		float nH = tNode.h_AStar + tNode.nodeCost + fNode.cost_transition_to_child[c] 
				+ fNode.cost_moveLimb_to_child[c] + getCostToGoal(fNode.hold_ids);
		return nH;
	}

	void updateHeuristicsForNode(int _fNode, int _tNode)
	{
		std::vector<int> mUpdateList;
		mUpdateList.reserve(stance_nodes.size() + 1);

		//
		mUpdateList.push_back(_fNode);
		while (mUpdateList.size() > 0)
		{
			int cNode = mUpdateList[0];
			mUpdateList.erase(mUpdateList.begin());

			mStanceNode& cStanceNode = stance_nodes[cNode];
			float min_val = FLT_MAX;
			int min_index = -1;
			for (unsigned int c = 0; c < cStanceNode.childStanceIds.size(); c++)
			{
				mStanceNode& childStanceNode = stance_nodes[cStanceNode.childStanceIds[c]];
				float cCost = getHeuristicCostNodeToNode(cNode, cStanceNode.childStanceIds[c]);
				if (min_val > cCost)
				{
					min_val = cCost;
					min_index = c;
				}
			}
			cStanceNode.bChildStanceIndex = cStanceNode.childStanceIds[min_index];
			cStanceNode.h_AStar = min_val;

			for (unsigned int f = 0; f < cStanceNode.parentStanceIds.size(); f++)
			{
				mStanceNode& fatherStanceNode = stance_nodes[cStanceNode.parentStanceIds[f]];
				if (fatherStanceNode.bChildStanceIndex == cNode)
				{
					mUpdateList.push_back(cStanceNode.parentStanceIds[f]);
				}
			}
		}
		// update best father index

	}

	/////////////////////////////////////////////////////////////////////////////////////////////////// build graph is not complete: needs debugging

	void buildGraph(std::vector<int>& iCStance, std::vector<int>& iInitStance) 
	{
		high_resolution_clock::time_point _t1 = high_resolution_clock::now();

		initialStance = iInitStance;

		if (mClimberSampler == nullptr)
		{
			return;
		}
		if (mClimberSampler->myHoldPoints.size() == 0)
		{
			return;
		}
		mRootGraph = addGraphNode(-1, iCStance);
		
		std::vector<int> expandNodes;
		expandNodes.push_back(mRootGraph);

		//add all nodes connected to initial stance.
		//PERTTU: Todo: what do we need the while for? Isn't it enough to add the mInitialList once, as getListOfStanceSamples() always gets the iInitStance as argument?
		while(expandNodes.size() > 0)
		{
			int exanpd_stance_id = expandNodes[0];
			expandNodes.erase(expandNodes.begin());
			stance_nodes[exanpd_stance_id].isItExpanded = true;

			std::vector<std::vector<int>> mInitialList = mClimberSampler->getListOfStanceSamples(stance_nodes[exanpd_stance_id].hold_ids, iInitStance, true);

			for (unsigned int i = 0; i < mInitialList.size(); i++)
			{
				int sId = addGraphNode(exanpd_stance_id, mInitialList[i]);

				if (!stance_nodes[sId].isItExpanded && !mSampler::isInSetHoldIDs(sId, expandNodes))
				{
					expandNodes.push_back(sId);
				}
			}
		}
		
		/*expandNodes.push_back(findStanceFrom(iInitStance));

		while(expandNodes.size() > 0)
		{
			int exanpd_stance_id = expandNodes[0];
			expandNodes.erase(expandNodes.begin());
			stance_nodes[exanpd_stance_id].isItExpanded = true;

			std::vector<std::vector<int>> all_possible_stances = mClimberSampler->getListOfStanceSamplesAround(stance_nodes[exanpd_stance_id].hold_ids, false);

			for (unsigned int ws = 0; ws < all_possible_stances.size(); ws++)
			{
				std::vector<std::vector<int>> mInitialList = mClimberSampler->getListOfStanceSamples(stance_nodes[exanpd_stance_id].hold_ids, all_possible_stances[ws], false);

				for (unsigned int i = 0; i < mInitialList.size(); i++)
				{
					int sId = addGraphNode(exanpd_stance_id, mInitialList[i]);

					if (!stance_nodes[sId].isItExpanded && !mSampler::isInSetHoldIDs(sId, expandNodes))
					{
						expandNodes.push_back(sId);
					}
				}
			}
		}*/

		int fStance = findStanceFrom(iInitStance);

		//std::vector<int> dStance;
		//dStance.push_back(1);
		//dStance.push_back(1);
		//dStance.push_back(2);
		//dStance.push_back(3);

		expandNodes.push_back(fStance);
		std::vector<int> reachable_holds_ids;
		

		//add other nodes
		while(expandNodes.size() > 0)
		{
			int exanpd_stance_id = expandNodes[0];
			expandNodes.erase(expandNodes.begin());
			stance_nodes[exanpd_stance_id].isItExpanded = true;

			for (unsigned int i = 0; i < stance_nodes[exanpd_stance_id].hold_ids.size(); i++)
			{
				std::vector<int> higher_stance_ids;
				if (stance_nodes[exanpd_stance_id].hold_ids[i] != -1)
				{
					if (i <= 1)
					{
						// for feet add all reachable holds above them
						higher_stance_ids = mClimberSampler->indices_higher_than[stance_nodes[exanpd_stance_id].hold_ids[i]];
					}
					else
					{
						// add hands for finding all combinations under them
						mSampler::addToSetHoldIDs(stance_nodes[exanpd_stance_id].hold_ids[i], reachable_holds_ids);
					}
				}
				for (unsigned int m = 0; m < higher_stance_ids.size(); m++)
				{
					mSampler::addToSetHoldIDs(higher_stance_ids[m], reachable_holds_ids);
				}
			}

			std::vector<std::vector<int>> all_possible_stances;
			for (unsigned int j = 0; j < reachable_holds_ids.size(); j++)
			{
				int hold_id_i = reachable_holds_ids[j];

				mClimberSampler->getListOfStanceSamplesAround(hold_id_i, stance_nodes[exanpd_stance_id].hold_ids,all_possible_stances);

				for (unsigned int ws = 0; ws < all_possible_stances.size(); ws++)
				{
					/*if (mNode::isSetAEqualsSetB(dStance, all_possible_stances[ws]) && fStance == exanpd_stance_id)
					{
						int notifyme = 1;
					}*/

					//std::vector<std::vector<int>> mInitialList = mClimberSampler->getListOfStanceSamples(stance_nodes[exanpd_stance_id].hold_ids, all_possible_stances[ws], false);

					//for (unsigned int i = 0; i < mInitialList.size(); i++)
					//{
					int sId = addGraphNode(exanpd_stance_id, all_possible_stances[ws]);//mInitialList[i]);

					if (!stance_nodes[sId].isItExpanded && !mSampler::isInSetHoldIDs(sId, expandNodes))
					{
						expandNodes.push_back(sId);
					}
					//}
				}
			}
			reachable_holds_ids.clear();
		}

//		applyDijkstraAll2(); // used for admisibale A*
		minmax.init(stance_nodes.size());
		printf("%d", stance_nodes.size());

		//for (volatile int j=0; j<10000; j++)
		//	applyDijkstraAll2();

		high_resolution_clock::time_point _t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(_t2 - _t1);
		timeOfCreation = (int)(time_span.count()*1000.0);

		return;
	}

	bool updateGraphNN(int _fNode, int _tNode)//, bool _retTrueForChange = true)
	{
		for (unsigned int i = 0; i < stance_nodes[_fNode].childStanceIds.size(); i++)
		{
			if (stance_nodes[_fNode].childStanceIds[i] == _tNode)
			{
			//	if (isDynamicGraph)
			//	{
				stance_nodes[_fNode].cost_transition_to_child[i] += _MaxTriedTransitionCost;
//				updateHeuristicsForNode(_fNode, _tNode);
				return true;
			//	}
				//if (_retTrueForChange)
				//{
				//	return true;
				//}
				//else
				//{
				//	return false;
				//}
			}
		}
		return false;
	}

	/*bool updateGraphPath(int _from, int pathSize)
	{
		for (int i = _from; i < pathSize; i++)
		{
			updateGraphNN(getIndexStanceNode(i-1), getIndexStanceNode(i));
		}
		return false;
	}*/
	
	std::vector<int> solveGraph(float& ret_min_val, mAlgSolveGraph algID)
	{
		std::vector<int> ret_path;
		switch (algID)
		{
		case mAlgSolveGraph::myAlgAStar:
			ret_path = solveAStar(ret_min_val);
			break;
		case mAlgSolveGraph::AStarPrune:
			ret_path = solveAStarPrune(ret_min_val);
			break;
		case mAlgSolveGraph::myAStarPrune:
			ret_path = solveMYAStarPrune(ret_min_val);
			break;
		case mAlgSolveGraph::myDijkstraHueristic:
			ret_path = solveAStarPrune2(ret_min_val);
			break;
		default:
			ret_path = solveAStar(ret_min_val);
			break;
		}
		return ret_path;
	}

	std::vector<std::vector<int>> returnPath()
	{
		return returnPathFrom(retPath);
	}

	int findStanceFrom(const std::vector<int>& _fStance)
	{
		uint32_t key=stanceToKey(_fStance);
		std::unordered_map<uint32_t,int>::iterator it=stanceToNode.find(key);
		if (it==stanceToNode.end())
			return -1;
		return it->second;
		//int stance_id = -1;
		//for (unsigned int i = 0; i < stance_nodes.size() && stance_id == -1; i++)
		//{
		//	if (mNode::isSetAEqualsSetB(stance_nodes[i].hold_ids, _fStance))
		//	{
		//		stance_id = i;
		//	}
		//}
		//return stance_id;
	}

	std::vector<int> getStanceGraph(int _graphNodeId)
	{
		return stance_nodes[_graphNodeId].hold_ids;
	}

	int getIndexStanceNode(int indexPath)
	{
		int j = retPath.size() - 1 - indexPath;
		if (j < (int)retPath.size())
		{
			return retPath[j];
		}
		return mRootGraph;
	}

	void setCurPathTriedFather()
	{
		for (unsigned int i = 0; i < retPath.size(); i++)
		{
			setTriedToCurrentFather(retPath[i], stance_nodes[retPath[i]].bFatherStanceIndex, true);
			setTriedToCurrentChild(stance_nodes[retPath[i]].bFatherStanceIndex, retPath[i], true);
		}
		return;
	}

	std::vector<int> retPath;

	//float preVal;
	//std::vector<int> preRetPath;

	std::list<mStancePathNode> openListPath;

	mSampler* mClimberSampler;

	Vector3 goalPos;
private:
	///////////////////////////// my alg A* ////////////////////////////////////
	std::vector<int> solveAStar(float& ret_min_val)
	{
		std::vector<int> openList;
		std::vector<int> closeList;

		stance_nodes[mRootGraph].g_AStar = 0;
		stance_nodes[mRootGraph].h_AStar = 0;

		openANode(mRootGraph, openList, closeList);

		int sIndex = findStanceFrom(initialStance);

		bool flag_continue = true;
		while (flag_continue)
		{
			int openIndexi = stanceIndexLowestFValue(openList, ret_min_val);

			if (isGoalFound(openIndexi))
			{
				retPath = returnPathFrom(openIndexi);
				if (mSampler::isInSetHoldIDs(sIndex, retPath))
				{
					if (!mSampler::isInSampledStanceSet(retPath, m_found_paths))
					{
						flag_continue = false;
					}
					else
					{
						mSampler::removeFromSetHoldIDs(openIndexi, openList);
						continue;
					}
				}
				else
				{
					retPath.clear();
				}
			}

			if (flag_continue && openIndexi >= 0)
			{
				openANode(openIndexi, openList, closeList);
			}

			if (openList.size() == 0)
			{
				flag_continue = false;
			}
		}

		/*if (preRetPath.size() == 0 || preVal <= ret_min_val)
		{
			preRetPath = retPath;
			preVal = ret_min_val;
		}*/
		/*else
		{
			float v;
			solveAStar(v);
		}*/
		return retPath;
	}

	void openANode(int gNodei, std::vector<int>& openList, std::vector<int>& closeList)
	{
		mSampler::removeFromSetHoldIDs(gNodei, openList);
		mSampler::addToSetHoldIDs(gNodei, closeList);

		mStanceNode n_i = stance_nodes[gNodei];
		for (unsigned int i = 0; i < n_i.childStanceIds.size(); i++)
		{
			int cIndex = n_i.childStanceIds[i];
			mStanceNode* n_child_i = &stance_nodes[cIndex];

			if (mSampler::isInSetHoldIDs(cIndex, closeList))
			{
				continue;
			}

			int oBestFather = n_child_i->bFatherStanceIndex;

			float nG = n_i.g_AStar + n_i.cost_transition_to_child[i] + n_i.cost_moveLimb_to_child[i] + n_child_i->nodeCost;
			float nH = getCostToGoal(n_child_i->hold_ids);
			float nF = nG + nH;

			if (!mSampler::isInSetHoldIDs(cIndex, openList))
			{
				float oG = n_child_i->g_AStar;

				n_child_i->g_AStar = nG;
				n_child_i->h_AStar = nH;
				n_child_i->bFatherStanceIndex = gNodei;
				n_child_i->childIndexInbFather = i;

				if (isLoopCreated(cIndex))
				{
					n_child_i->bFatherStanceIndex = oBestFather;
					n_child_i->g_AStar = oG;
				}

				mSampler::addToSetHoldIDs(cIndex, openList);
			}
			else
			{
				float oG = n_child_i->g_AStar;

				if (nG < oG || (isItTriedToCurrentFather(n_child_i->stanceIndex, n_child_i->bFatherStanceIndex) && isAllOfNodeChildrenTried(n_child_i->stanceIndex)))
				{
					n_child_i->g_AStar = nG;
					n_child_i->h_AStar = nH;
					n_child_i->bFatherStanceIndex = gNodei;
					n_child_i->childIndexInbFather = i;

					if (isLoopCreated(cIndex))
					{
						n_child_i->bFatherStanceIndex = oBestFather;
						n_child_i->g_AStar = oG;
					}
				}
			}
		}

		return;
	}

	int stanceIndexLowestFValue(std::vector<int>& openList, float& ret_min_val)
	{
		float minVal_notTried = FLT_MAX;
		int min_index_notTried = -1;

		float minVal = FLT_MAX;
		int min_index = -1;
		for (unsigned int i = 0; i < openList.size(); i++)
		{
			//mStanceNode* cFather = nullptr;
			//if (stance_nodes[openList[i]].bFatherStanceIndex != -1)
			//{
			//	cFather = &stance_nodes[stance_nodes[openList[i]].bFatherStanceIndex];
			//}
			mStanceNode* cNode = &stance_nodes[openList[i]];

			float cVal = stance_nodes[openList[i]].g_AStar + stance_nodes[openList[i]].h_AStar;

			if (cVal < minVal_notTried && (!isItTriedToCurrentFather(cNode->stanceIndex, cNode->bFatherStanceIndex) || !isAllOfNodeChildrenTried(openList[i])))
			{
				minVal_notTried = cVal;
				min_index_notTried = openList[i];
			}

			if (cVal < minVal)
			{
				minVal = cVal;
				min_index = openList[i];
			}
		}

		ret_min_val = minVal_notTried;

		if (min_index_notTried >= 0)
		{
			return min_index_notTried;
		}

		ret_min_val = minVal;
		return min_index;
	}

	///////////////////////////// A*prune /////////////////////////////////////

	void applyDijkstra(int fIndex, int sIndex)
	{
		for (unsigned int i = 0; i < stance_nodes.size(); i++)
		{
			stance_nodes[i].g_AStar = FLT_MAX;
		}

		std::vector<int> openList;
		std::vector<int> closeList;

		stance_nodes[fIndex].g_AStar = 0;
		openList.push_back(fIndex);
		
		while (openList.size() > 0)
		{
			mStanceNode* cNode = &stance_nodes[openList[0]];
			openList.erase(openList.begin());
			mSampler::addToSetHoldIDs(cNode->stanceIndex, closeList);

			if (isGoalFound(cNode->stanceIndex))
			{
				retPath = returnPathFrom(cNode->stanceIndex);
				stance_nodes[fIndex].h_AStar = cNode->g_AStar;
				return;
			}

			for (unsigned int c = 0; c < cNode->childStanceIds.size(); c++)
			{
				mStanceNode* ccNode = &stance_nodes[cNode->childStanceIds[c]];
				float nG = cNode->g_AStar + cNode->cost_transition_to_child[c] + cNode->cost_moveLimb_to_child[c] + ccNode->nodeCost;
				if (nG < ccNode->g_AStar)
				{
					ccNode->g_AStar = nG;
					ccNode->bFatherStanceIndex = cNode->stanceIndex;
				}

				if (!mSampler::isInSetHoldIDs(ccNode->stanceIndex, closeList))
				{
					if (mSampler::isInSetHoldIDs(ccNode->stanceIndex, openList))
					{
						mSampler::removeFromSetHoldIDs(ccNode->stanceIndex, openList);
					}
					unsigned int i = 0;
					for (i = 0; i < openList.size(); i++)
					{
						float cF = stance_nodes[openList[i]].g_AStar;
						float nF = nG;
						if (nF < cF)
						{
							break;
						}
					}
					openList.insert(openList.begin() + i, ccNode->stanceIndex);
				}
			}
		}

		return;
	}

	std::vector<int> solveAStarPrune(float& ret_min_val)
	{
		int sIndex = findStanceFrom(initialStance);
		for (unsigned int i = 0; i < stance_nodes.size(); i++)
		{
			if (!isGoalFound(stance_nodes[i].stanceIndex))
				applyDijkstra(stance_nodes[i].stanceIndex, sIndex);
			else
				stance_nodes[i].h_AStar = 0.0f;
		}

		std::vector<mStancePathNode> openListPath;

		std::vector<int> rootPath; rootPath.push_back(mRootGraph);
		openListPath.push_back(mStancePathNode(rootPath,0,0));

		bool flag_continue = true;
		while (flag_continue)
		{
			mStancePathNode fNode = openListPath[0];
			openListPath.erase(openListPath.begin());

			int eStanceIndex = fNode._path[fNode._path.size()-1];

			if (isGoalFound(eStanceIndex))
			{
				retPath = reversePath(fNode._path);
				ret_min_val = fNode.mG + fNode.mH;
				if (mSampler::isInSetHoldIDs(sIndex, retPath))
				{
					if (!mSampler::isInSampledStanceSet(retPath, m_found_paths))
					{
						flag_continue = false;
					}
				}
				else
				{
					retPath.clear();
				}
			}

			if (flag_continue)
			{
				mStanceNode stanceNode = stance_nodes[eStanceIndex];
				for (unsigned int c = 0; c < stanceNode.childStanceIds.size(); c++)
				{
					mStanceNode stanceChild = stance_nodes[stanceNode.childStanceIds[c]];
				
					if (mSampler::isInSetHoldIDs(stanceChild.stanceIndex, fNode._path)) // no loop
						continue;

					float nG = fNode.mG + stanceNode.cost_transition_to_child[c] + stanceNode.cost_moveLimb_to_child[c] + stanceChild.nodeCost;
					float nH = stanceChild.h_AStar;

					std::vector<int> nPath = fNode.addToEndPath(stanceChild.stanceIndex);

					mStancePathNode nStancePath(nPath, nG, nH);

					insertNodePath(openListPath, nStancePath);
				}
			}

			if (openListPath.size() == 0)
			{
				flag_continue = false;
			}
		}

		return retPath;
	}

	void insertNodePath(std::vector<mStancePathNode>& openList, mStancePathNode& nNode)
	{
		unsigned int i = 0;
		for (i = 0 ; i < openList.size(); i++)
		{
			mStancePathNode n_i = openList[i];
			float cF = n_i.mG + n_i.mH;
			float nF = nNode.mG + nNode.mH;
			if (nF < cF)
			{
				break;
			}
		}
		openList.insert(openList.begin() + i, nNode);
		return;
	}

	////////////////////////////////////////////////////////////// A* prune my version

	std::vector<int> solveMYAStarPrune(float& ret_min_val)
	{
		int max_count = 200000;
		int sIndex = findStanceFrom(initialStance);

		std::list<mStancePathNode> openListPath;
		std::list<mStancePathNode> closeListPath;

		std::vector<int> openList; openList.reserve(max_count);
		std::vector<int> closeList; closeList.reserve(max_count);

		std::vector<int> rootPath; rootPath.push_back(mRootGraph);
		openList.push_back(mRootGraph);
		openListPath.push_back(mStancePathNode(rootPath,0,0));

		bool flag_continue = true;
		while (flag_continue)
		{
			if (openListPath.size() == 0)
			{
				movePathsFromTo(closeListPath, openListPath, openList, closeList);
			}

			if (openListPath.size() == 0)
			{
				break;
			}

			mStancePathNode fNode = openListPath.front();
			openListPath.pop_front();// erase(openListPath.begin());

			int eStanceIndex = fNode._path[fNode._path.size()-1];

			if (isGoalFound(eStanceIndex))
			{
				retPath = reversePath(fNode._path);
				ret_min_val = fNode.mG + fNode.mH;
				if (mSampler::isInSetHoldIDs(sIndex, retPath))
				{
					if (!mSampler::isInSampledStanceSet(retPath, m_found_paths))
					{
						flag_continue = false;
					}
					else
					{
						mSampler::removeFromSetHoldIDs(eStanceIndex, openList);
						continue;
					}
				}
				else
				{
					retPath.clear();
				}
			}

			if (flag_continue)
			{
				mStanceNode& stanceNode = stance_nodes[eStanceIndex];

				mSampler::removeFromSetHoldIDs(eStanceIndex, openList);
				mSampler::addToSetHoldIDs(eStanceIndex, closeList);

				for (unsigned int c = 0; c < stanceNode.childStanceIds.size(); c++)
				{
					mStanceNode& stanceChild = stance_nodes[stanceNode.childStanceIds[c]];
				
					if (mSampler::isInSetHoldIDs(stanceChild.stanceIndex, fNode._path)) // no loop
						continue;

					float nG = fNode.mG + stanceNode.cost_transition_to_child[c] + stanceNode.cost_moveLimb_to_child[c] + stanceChild.nodeCost;
					float nH = getCostToGoal(stanceChild.hold_ids);

					std::vector<int> nPath = fNode.addToEndPath(stanceChild.stanceIndex);

					mStancePathNode nStancePath(nPath, nG, nH);

					insertNodePath(openListPath, closeListPath, nStancePath, openList, closeList);
				}
			}

			if (openListPath.size() == 0 && closeListPath.size() == 0)
			{
				flag_continue = false;
			}
		}

		return retPath;
	}

	void movePathsFromTo(std::list<mStancePathNode>& closeListPath, std::list<mStancePathNode>& openListPath, std::vector<int>& openListID, std::vector<int>& closeListID)
	{
		unsigned int max_open_list_size = 100;
		std::list<mStancePathNode>::iterator iter_pathNode = closeListPath.begin();
		int max_counter = closeListPath.size();
		closeListID.clear();
		for (int i = 0; i < max_counter; i++)
		{
			const mStancePathNode& nPath = *iter_pathNode;
			int eIndex = nPath._path[nPath._path.size() - 1];

//			mSampler::removeFromSetHoldIDs(eIndex, closeListID);

			mStancePathNode nNode = *iter_pathNode;
			iter_pathNode = closeListPath.erase(iter_pathNode);
			insertNodePath(openListPath, closeListPath, nNode, openListID, closeListID);

			if (openListID.size() > max_open_list_size)
			{
				return;
			}

			if (iter_pathNode == closeListPath.end())
				break;
		}
		return;
	}

	void insertNodePath(std::list<mStancePathNode>& openListPath, std::list<mStancePathNode>& closeListPath, mStancePathNode& nNode, std::vector<int>& openListID, std::vector<int>& closeListID)
	{
		unsigned int fIndex = 0;
		unsigned int eIndex = 0;

		int cIndex = nNode._path[nNode._path.size()-1];
		if (mSampler::isInSetHoldIDs(cIndex, closeListID))
		{
			closeListPath.push_back(nNode);
			return;
		}
		else
		{
			if (!mSampler::isInSetHoldIDs(cIndex, openListID))
			{
				eIndex = openListID.size();
				mSampler::addToSetHoldIDs(cIndex, openListID);
			}
			else
			{
				
				std::list<mStancePathNode>::iterator iter_pathNode = openListPath.begin();
				for (unsigned int i = 0; i < openListID.size(); i++)
				{
					const mStancePathNode& n_i = *iter_pathNode;

					int endIndex = n_i._path[n_i._path.size()-1];
					if (endIndex == cIndex)
					{
						float cF = n_i.mG + n_i.mH;
						float nF = nNode.mG + nNode.mH;
						if (nF < cF)
						{
							mStancePathNode copy_n_i = *iter_pathNode;
							iter_pathNode = openListPath.erase(iter_pathNode);
							insertPathNode(openListPath, nNode, 0, openListID.size() - 1);
							closeListPath.push_back(copy_n_i);
							return;
						}
						else
						{
							closeListPath.push_back(nNode);
							return;
						}
					}
					iter_pathNode++;
				}
			}
			
		}
		insertPathNode(openListPath, nNode, fIndex, eIndex);
		return;
	}

	void insertPathNode(std::list<mStancePathNode>& openListPath, const mStancePathNode& nNode, unsigned int fIndex, unsigned int eIndex)
	{
		
		std::list<mStancePathNode>::iterator iter = openListPath.begin();

		//while (iter != openListPath.end()){
		//	//CODE HERE
		//	iter++;
		//}

		unsigned int i = fIndex;
		for (i = fIndex; i < eIndex; i++)
		{
			const mStancePathNode& n_i = *iter;
			float cF = n_i.mG + n_i.mH;
			float nF = nNode.mG + nNode.mH;
			if (nF < cF)
			{
				break;
			}
			iter++;
		}
		openListPath.insert(iter, nNode);
	}

	//////////////////////////////////////////////////////////////////
	/*std::vector<int> solveDijkstraKPath(float& ret_min_val)
	{
		int max_num_paths = 60;



		std::vector<int> ret_path;
		return ret_path;
	}*/

	int getChildIndex(int _fNode, int _tNode)
	{
		for (unsigned int i = 0; i < stance_nodes[_fNode].childStanceIds.size(); i++)
		{
			if (stance_nodes[_fNode].childStanceIds[i] == _tNode)
			{
				return i;
			}
		}
		return -1;
	}

	void applyDijkstraAll2()
	{
		//Run Dijkstra backwards, initializing all goal nodes to zero cost and others to infinity.
		//This results in each node having the cost as the minimum cost to go towards any of the goals.
		//This will then be used as the heuristic in A* prune. Note that if the climber is able to make
		//all the moves corresponding to all edges, the heuristic equals the actual cost, and A* will be optimal.
		//In case the climber fails, this function will be called again, i.e., the heuristic will be updated
		//after each failure.
		//Note: minmax is a DynamicMinMax instance - almost like std::priority_queue, but supports updating 
		//the priorities of elements at a logN cost without removal & insertion. 
		for (unsigned int i = 0; i < stance_nodes.size(); i++)
		{
			stance_nodes[i].g_AStar = FLT_MAX;   
			stance_nodes[i].h_AStar = FLT_MAX;  //this is the output value, i.e., the "heuristic" used by A* prune
			stance_nodes[i].dijkstraVisited=false;
			minmax.setValue(i,FLT_MAX);
		}
		//for (unsigned int i = 0; i < reached_goals.size(); i++)
		//{
		//	openList.push_back(reached_goals[i]);
		//	stance_nodes[reached_goals[i]].h_AStar = 0;
		//	minmax.setValue(reached_goals[i],0);
		//}
		for (unsigned int i = 0; i < goal_stances.size(); i++)
		{
			stance_nodes[goal_stances[i]].h_AStar = 0;
			minmax.setValue(goal_stances[i],0);
		}

	//	int sIndex = findStanceFrom(initialStance);
		int nVisited=0;
		while (nVisited<(int)stance_nodes.size())
		{
			//get the node with least cost (initially, all goal nodes have zero cost)
			mStanceNode* cNode =&stance_nodes[minmax.getMinIdx()];
			
			//loop over neighbors (parentStanceIds really contains all the neighbors)
			//and update their costs
			for (unsigned int f = 0; f < cNode->parentStanceIds.size(); f++)
			{
				mStanceNode* fNode = &stance_nodes[cNode->parentStanceIds[f]];
				int c = getChildIndex(cNode->parentStanceIds[f], cNode->stanceIndex);
				float nH = cNode->h_AStar + cNode->nodeCost + fNode->cost_transition_to_child[c] 
						+ fNode->cost_moveLimb_to_child[c];// + getDisFromStanceToStance(*fNode, *cNode);
							
				//getCostToGoal(fNode->hold_ids);  //TODO fix getCostToGoal! (should equal the minimum distance from left or right hand to goal, and the same should be used as the actual cost).
				
						/*float cDis = 0.0f;
				for (unsigned int h_i = 0; h_i < fNode->hold_ids.size(); h_i++)
				{
					if (fNode->hold_ids[h_i] != -1 && cNode->hold_ids[h_i] != -1)
					{
						cDis+=(mClimberSampler->getHoldPos(fNode->hold_ids[h_i]) - mClimberSampler->getHoldPos(cNode->hold_ids[h_i])).squaredNorm();
					}
				}
				nH += sqrt(cDis);*/


				//PERTTU: commented this check out - makes no sense. We should update all neighbors except the already visited ones
				//if (cNode->bFatherStanceIndex == cNode->parentStanceIds[f] || fNode->bFatherStanceIndex >= 0)
				if (!fNode->dijkstraVisited)
				{
					if (nH < fNode->h_AStar)
					{
						fNode->h_AStar = nH;
						fNode->bChildStanceIndex = cNode->stanceIndex;
						minmax.setValue(fNode->stanceIndex,nH);
					}
				}
			} //all neighbors
			//Mark the node as visited. Also set it's priority to FLT_MAX so that it will not be returned by minmax.getMinIdx().
			//Note that since dijkstraVisited is now true, the priority will not be updated again and will stay at FLT_MAX for the remainder of the algorithm.
			cNode->dijkstraVisited=true;
			minmax.setValue(cNode->stanceIndex,FLT_MAX);
			nVisited++;
		}
		
		return;
	}

	std::vector<int> solveAStarPrune2(float& ret_min_val)
	{
		high_resolution_clock::time_point _t1 = high_resolution_clock::now();

		int k = 100; // k shortest path in the paper
		int max_number_paths = max<int>(maxGraphDegree + 20, k);

		int sIndex = findStanceFrom(initialStance);

		applyDijkstraAll2();

		bool flag_continue = true;
		if (openListPath.size() == 0)
		{
			flag_continue = false;
			retPath.clear();
		}
		while (flag_continue)
		{
			mStancePathNode fNode = openListPath.front();
			openListPath.pop_front();// erase(openListPath.begin());

			int eStanceIndex = fNode._path[fNode._path.size()-1];

			if (isGoalFound(eStanceIndex))
			{
				retPath = reversePath(fNode._path);
				ret_min_val = fNode.mG + fNode.mH;
				if (mSampler::isInSetHoldIDs(sIndex, retPath))
				{
					if (!mSampler::isInSampledStanceSet(retPath, m_found_paths))
					{
						flag_continue = false;
					}
					else
					{
						continue;
					}
				}
				else
				{
					retPath.clear();
				}
			}

			if (flag_continue)
			{
				mStanceNode& stanceNode = stance_nodes[eStanceIndex];
				for (unsigned int c = 0; c < stanceNode.childStanceIds.size(); c++)
				{
					mStanceNode& stanceChild = stance_nodes[stanceNode.childStanceIds[c]];

					if (mSampler::isInSetHoldIDs(stanceChild.stanceIndex, fNode._path)) // no loop
						continue;

					float nG = fNode.mG + stanceNode.cost_transition_to_child[c] + stanceNode.cost_moveLimb_to_child[c] + stanceChild.nodeCost;// + getDisFromStanceToStance(stanceNode, stanceChild);
					float nH = stanceChild.h_AStar;

					std::vector<int> nPath = fNode.addToEndPath(stanceChild.stanceIndex);

					mStancePathNode nStancePath(nPath, nG, nH);

					insertNodePath(openListPath, nStancePath, max_number_paths);
				}
			}

			if (openListPath.size() == 0)
			{
				flag_continue = false;
			}
		}

		high_resolution_clock::time_point _t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(_t2 - _t1);
		if (retPath.size() != 0)
			timeOfFindingPaths.push_back((int)(time_span.count()*1000.0));

		return retPath;
	}

	void insertNodePath(std::list<mStancePathNode>& openList, mStancePathNode& nNode, int max_num)
	{
		int k = 0;
		std::list<mStancePathNode>::iterator iter_pathNode = openList.begin();
		while (iter_pathNode != openList.end())
		{
			mStancePathNode& n_i = *iter_pathNode;//openList[i];
			float cF = n_i.mG + n_i.mH;
			float nF = nNode.mG + nNode.mH;
			if (nF < cF)
			{
				break;
			}
			iter_pathNode++;
			k++;
		}
		if (k < max_num)
			openList.insert(iter_pathNode, nNode);

		if ((int)openList.size() > max_num)
		{
			openList.pop_back();
		}
		return;
	}
	//////////////////////////////////////////////////////////////

	std::vector<int> reversePath(std::vector<int>& _iPath)
	{
		std::vector<int> nPath;
		for (int i = _iPath.size() - 1; i >=0; i--)
		{
			nPath.push_back(_iPath[i]);
		}
		return nPath;
	}

	//////////////////////////////////////////////////////// the huristic values to goal is not admisible
	float getCostToGoal(std::vector<int>& iCStance)
	{
		/*float cCost = 0.0f; 
		for (unsigned int i = 0; i < iCStance.size(); i++)
		{ 
			if (iCStance[i] != -1)
			{
				cCost += (mClimberSampler->getHoldPos(iCStance[i]) - goalPos).squaredNorm();
			}
			else
			{
				cCost += 0.0f;
			}
		}
		return 0.01f * sqrt(cCost);*/
		float cCount = 0.0f;
		std::vector<Vector3> sample_desired_hold_points;
		Vector3 midPoint = mClimberSampler->getHoldStancePosFrom(iCStance, sample_desired_hold_points, cCount);
		return (midPoint - goalPos).norm();
	}
	//public:
	float getCostAtNode(std::vector<int>& iCStance, bool printDebug=false)
	{
		float k_crossing = 100;
		float k_hanging_hand = 200 + 50; // 200
		float k_hanging_leg = 10 + 10; // 10
//		float k_hanging_more_than2 = 0;//100;
		float k_matching = 100;
		float k_dis = 1000;

		float _cost = 0.0f;

		// punish for hanging more than one limb
		int counter_hanging = 0;
		for (unsigned int i = 0; i < iCStance.size(); i++)
		{
			if (iCStance[i] == -1)
			{
				counter_hanging++;
				
				if (i >= 2)
				{
					// punish for having hanging hand
					_cost += k_hanging_hand;
					if (printDebug) rcPrintString("Hanging hand!");
				}
				else
				{
					// punish for having hanging hand
					_cost += k_hanging_leg;
				}
			}
		}

		//// punish for having two or more limbs hanging
		//if (counter_hanging >= 2) 
		//{
		//	_cost += k_hanging_more_than2;
		//	if (printDebug) rcPrintString("More than 2 hanging limbs!");
		//}

		// crossing hands
		if (iCStance[2] != -1 && iCStance[3] != -1)
		{
			Vector3 rHand = mClimberSampler->getHoldPos(iCStance[3]);
			Vector3 lHand = mClimberSampler->getHoldPos(iCStance[2]);

			if (rHand.x() < lHand.x())
			{
				_cost += k_crossing;
				if (printDebug) rcPrintString("Hands crossed!");
			}
		}

		// crossing feet
		if (iCStance[0] != -1 && iCStance[1] != -1)
		{
			Vector3 lLeg = mClimberSampler->getHoldPos(iCStance[0]);
			Vector3 rLeg = mClimberSampler->getHoldPos(iCStance[1]);

			if (rLeg.x() < lLeg.x())
			{
				_cost += k_crossing;
				if (printDebug) rcPrintString("Legs crossed!");
			}
		}

		// crossing hand and foot
		for (unsigned int i = 0; i <= 1; i++)
		{
			if (iCStance[i] != -1)
			{
				Vector3 leg = mClimberSampler->getHoldPos(iCStance[i]);
				for (unsigned int j = 2; j <= 3; j++)
				{
					if (iCStance[j] != -1)
					{
						Vector3 hand = mClimberSampler->getHoldPos(iCStance[j]);
						
						if (hand.z() <= leg.z())
						{
							_cost += k_crossing;
						}
					}
				}
			}
		}

		//feet matching
		if (iCStance[0]==iCStance[1])
		{
			_cost += k_matching;
			if (printDebug) rcPrintString("Feet matched!");
		}

		//punishment for hand and leg being close
		for (unsigned int i = 0; i <= 1; i++)
		{
			if (iCStance[i] != -1)
			{
				Vector3 leg = mClimberSampler->getHoldPos(iCStance[i]);
				for (unsigned int j = 2; j <= 3; j++)
				{
					if (iCStance[j] != -1)
					{
						Vector3 hand = mClimberSampler->getHoldPos(iCStance[j]);
						
						float cDis = (hand - leg).norm();
						
						const float handAndLegDistanceThreshold = 0.5f;//mClimberSampler->climberRadius / 2.0f;
						if (cDis < handAndLegDistanceThreshold)
						{
							cDis/=handAndLegDistanceThreshold;
							_cost += k_dis*std::max(0.0f,1.0f-cDis);
							if (printDebug) rcPrintString("Hand and leg too close! v = %f", k_dis*std::max(0.0f,1.0f-cDis));
						}
					}
				}
			}
		}

		return _cost;
	}
	//private:
	std::vector<Vector3> getExpectedPositionSigma(Vector3 midPoint)
	{
		float r = mClimberSampler->climberRadius;
		
		std::vector<float> theta_s;
		theta_s.push_back(PI + PI / 4.0f);
		theta_s.push_back(1.5 * PI + PI / 4.0f);
		theta_s.push_back(0.5 * PI + PI / 4.0f);
		theta_s.push_back(PI / 4.0f);

		std::vector<Vector3> expectedPoses;
		for (unsigned int i = 0; i < theta_s.size(); i++)
		{
			Vector3 iDir(cosf(theta_s[i]), 0.0f, sinf(theta_s[i]));
			expectedPoses.push_back(midPoint + (r / 2.0f) * iDir);
		}

		return expectedPoses;
	}

	float getDisFromStanceToStance(std::vector<int>& si, std::vector<int>& sj)
	{
		float cCount = 0.0f;
		std::vector<Vector3> hold_points_i;
		Vector3 midPoint1 = mClimberSampler->getHoldStancePosFrom(si, hold_points_i, cCount);
//		std::vector<Vector3> e_hold_points_i = getExpectedPositionSigma(midPoint1);

		std::vector<Vector3> hold_points_j;
		Vector3 midPoint2 = mClimberSampler->getHoldStancePosFrom(sj, hold_points_j, cCount);
//		std::vector<Vector3> e_hold_points_j = getExpectedPositionSigma(midPoint2);

		float cCost = 0.0f;
		float hangingLimbExpectedMovement=2.0f;
		for (unsigned int i = 0; i < si.size(); i++)
		{
			float coeff_cost = 1.0f;
			if (si[i] != sj[i])
			{
				Vector3 pos_i;
				if (si[i] != -1)
				{
					pos_i = hold_points_i[i];
				}
				else
				{
					//pos_i = e_hold_points_i[i];
					cCost += 0.5f;
					continue;
				}
				Vector3 pos_j;
				if (sj[i] != -1)
				{
					pos_j = hold_points_j[i];
				}
				else
				{
					//pos_j = e_hold_points_j[i];
					cCost += hangingLimbExpectedMovement;
					continue;
				}

				//favor moving hands
				if (i >= 2)
					coeff_cost = 0.9f;

				cCost += coeff_cost * (pos_i - hold_points_j[i]).squaredNorm();
			}
			else
			{
				if (sj[i] == -1)
				{
					cCost += hangingLimbExpectedMovement;
				}
			}
		}

		return sqrtf(cCost);
	}
	
	bool firstHoldIsLower(int hold1, int hold2)
	{
		if (hold1==-1 && hold2==-1)
			return false;
		if (hold1!=-1 && mClimberSampler->getHoldPos(hold1).z() < mClimberSampler->getHoldPos(hold2).z())
		{
			return true;
		}
		//first hold is "free" => we can't really know 
		return false;
	}

	float getCostMovementLimbs(std::vector<int>& si, std::vector<int>& sj)
	{
		float k_dis = 1.0f;
		float k_2limbs = 120.0f;//100.0f;
//		float k_freeAnother = 0.0f;//20.0f;
		float k_pivoting_close_dis = 500.0f;

		//First get the actual distance between holds. We scale it up 
		//as other penalties are not expressed in meters
		float _cost = k_dis * getDisFromStanceToStance(si, sj);

		//penalize moving 2 limbs, except in "ladder climbing", i.e., moving opposite hand and leg
		bool flag_punish_2Limbs = true;
		bool is2LimbsPunished = false;
		if (mNode::getDiffBtwSetASetB(si, sj) > 1.0f)
		{

			if (si[0] != sj[0] && si[3] != sj[3] && firstHoldIsLower(si[0],sj[0]))
			{
				flag_punish_2Limbs = false;
				if (sj[0] != -1 && sj[3] != -1 && mClimberSampler->getHoldPos(sj[3]).x() - mClimberSampler->getHoldPos(sj[0]).x() < 0.5f)
					flag_punish_2Limbs = true;
				if (sj[0] != -1 && sj[3] != -1 && mClimberSampler->getHoldPos(sj[3]).z() - mClimberSampler->getHoldPos(sj[0]).z() < 0.5f)
					flag_punish_2Limbs = true;
			}

			if (si[1] != sj[1] && si[2] != sj[2] && firstHoldIsLower(si[1],sj[1]))
			{
				flag_punish_2Limbs = false;
				if (sj[1] != -1 && sj[2] != -1 && mClimberSampler->getHoldPos(sj[1]).x() - mClimberSampler->getHoldPos(sj[2]).x() < 0.5f)
					flag_punish_2Limbs = true;
				if (sj[1] != -1 && sj[2] != -1 && mClimberSampler->getHoldPos(sj[2]).z() - mClimberSampler->getHoldPos(sj[1]).z() < 0.5f)
					flag_punish_2Limbs = true;
			}

			//if (si[0] == -1 && si[1] == -1)
			//{
			//	int notifyme = 1;
			//}
			if (flag_punish_2Limbs)
				_cost += k_2limbs;
		}
		/*else
		{
			if ((si[0] == -1 || si[1] == -1) && (si[2] == 2 && si[3] == 3))
			{
				int notifyme = 1;
			}
		}*/

		////penalize transitions where we will have both legs momentarily hanging 
		//if ((si[0]==-1 && (si[1] !=sj[1])) || (si[1]==-1 && (si[0] !=sj[0])))
		//{
		//	_cost+=150.0f;
		//}


		//One of the hand on a foot hold and the other hand is moved.
		//Commented out, as we are already punishing nodes where hand and foot are on the same hold
		//if (si[2] != sj[2] || si[3] != sj[3])
		//{
		//	if (si[2] != sj[2])
		//	{
		//		if ((si[3] == si[0] || si[3] == si[1]) && si[3] >= 0)
		//		{
		//			_cost += 100;
		//		}
		//	}
		//	if (si[3] != sj[3])
		//	{
		//		if ((si[2] == si[0] || si[2] == si[1]) && si[2] >= 0)
		//		{
		//			_cost += 100;
		//		}
		//	}
		//}

		// calculating the stance during the transition
		std::vector<int> sn(4);
		int count_free_limbs = 0;
		for (unsigned int i = 0; i < si.size(); i++)
		{
			if (si[i] != sj[i])
			{
				sn[i]=-1;
				count_free_limbs++;
			}
			else
			{
				sn[i] = si[i];
			}
		}
		// free another
		if (count_free_limbs >= 2 && mNode::getDiffBtwSetASetB(si, sj) == 1.0f)
			_cost += k_2limbs;

		// punish for pivoting!!!
		float v = 0.0f;
		float max_dis = -FLT_MAX;
		for (unsigned int i = 0; i <= 1; i++)
		{
			if (sn[i] != -1)
			{
				Vector3 leg = mClimberSampler->getHoldPos(sn[i]);
				for (unsigned int j = 2; j <= 3; j++)
				{
					if (sn[j] != -1)
					{
						Vector3 hand = mClimberSampler->getHoldPos(sn[j]);

						float cDis = (hand - leg).norm();

						if (max_dis < cDis)
							max_dis = cDis;
					}
				}
			}
		}
		if (max_dis >= 0 && max_dis < mClimberSampler->climberRadius / 2.0f && count_free_limbs > 1.0f)
		{
			v += k_pivoting_close_dis;
		}
		_cost += v;

		return _cost;
	}

	float getCostTransition(std::vector<int>& si, std::vector<int>& sj)
	{
		return 1.0f;
	}

	////////////////////////////////////////////////////////
	std::vector<int> returnPathFrom(int iStanceIndex)
	{
		std::vector<int> rPath;

		int cIndex = iStanceIndex;

		int counter = 0;
		while (cIndex >= 0) // cIndex == 0 is (-1,-1,-1,-1)
		{
			mStanceNode nSi = stance_nodes[cIndex];
			rPath.push_back(nSi.stanceIndex);

			cIndex = nSi.bFatherStanceIndex;

			counter++;

			if (counter > (int)stance_nodes.size())
				break;
		}

		return rPath;
	}

	std::vector<std::vector<int>> returnPathFrom(std::vector<int>& pathIndex)
	{
		std::vector<std::vector<int>> lPath;
		for (int i = pathIndex.size() - 1; i >= 0; i--)
		{
			mStanceNode nSi = stance_nodes[pathIndex[i]];
			lPath.push_back(nSi.hold_ids);
		}
		return lPath;
	}

	bool isGoalFound(int iStanceIndex)
	{
		return mSampler::isInSetHoldIDs(iStanceIndex, goal_stances);
	}

	void setTriedToCurrentFather(int cChildIndex, int cFather, bool val)
	{
		if (cFather < 0)
			return;
		for (unsigned int i = 0; i < stance_nodes[cChildIndex].parentStanceIds.size(); i++)
		{
			if (stance_nodes[cChildIndex].parentStanceIds[i] == cFather)
			{
				stance_nodes[cChildIndex].isItTried_to_father[i] = val;
				return;
			}
		}
		return;
	}

	void setTriedToCurrentChild(int cFatherIndex, int cChild, bool val)
	{
		if (cFatherIndex < 0 || cChild < 0)
			return;
		for (unsigned int i = 0; i < stance_nodes[cFatherIndex].childStanceIds.size(); i++)
		{
			if (stance_nodes[cFatherIndex].childStanceIds[i] == cChild)
			{
				stance_nodes[cFatherIndex].isItTried_to_child[i] = val;
				return;
			}
		}
		return;
	}

	bool isItTriedToCurrentFather(int cChildIndex, int cFather)
	{
		for (unsigned int i = 0; i < stance_nodes[cChildIndex].parentStanceIds.size(); i++)
		{
			if (stance_nodes[cChildIndex].parentStanceIds[i] == cFather)
			{
				return stance_nodes[cChildIndex].isItTried_to_father[i];
			}
		}
		return false;
	}
	
	bool isAllOfNodeChildrenTried(int cNodeIndex)
	{
		for (unsigned int i = 0; i < stance_nodes[cNodeIndex].isItTried_to_child.size(); i++)
		{
			if (!stance_nodes[cNodeIndex].isItTried_to_child[i])
			{
				return false;
			}
		}
		return true;
	}

	bool isLoopCreated(int iStanceIndex)
	{
		int cIndex = iStanceIndex;

		int counter = 0;
		while (cIndex != 0)
		{
			mStanceNode nSi = stance_nodes[cIndex];

			cIndex = nSi.bFatherStanceIndex;

			counter++;

			if (counter > (int)stance_nodes.size())
			{
				printf("Loooooooooooooooooooooooooooooop");
				return true;
			}
		}
		return false;
	}

	int addGraphNode(int _fromIndex, std::vector<int>& _sStance)
	{
	//	AALTO_ASSERT1(_sStance.size()==4);
		mStanceNode nStance(_sStance);
		nStance.stanceIndex = stance_nodes.size();

		if (_fromIndex == -1)
		{
			nStance.nodeCost = 0.0f;

			stance_nodes.push_back(nStance);
			int index = stance_nodes.size() - 1;
			stanceToNode[stanceToKey(nStance.hold_ids)]=index;
			return index;
		}

		int stance_id = findStanceFrom(nStance.hold_ids);

		if (stance_id == -1)
		{
			nStance.nodeCost = getCostAtNode(_sStance);

			if (nStance.hold_ids[3] != -1) // search for right hand
			{
				//printf("%d \n", nStance.hold_ids[3]);
				Vector3 holdPos = mClimberSampler->getHoldPos(nStance.hold_ids[3]);
				if ((holdPos - goalPos).norm() < 0.1f)
				{
					mSampler::addToSetHoldIDs(nStance.stanceIndex, goal_stances);
				}
			}
			if (nStance.hold_ids[2] != -1) // search for right hand
			{
				//printf("%d \n", nStance.hold_ids[2]);
				Vector3 holdPos = mClimberSampler->getHoldPos(nStance.hold_ids[2]);
				if ((holdPos - goalPos).norm() < 0.1f)
				{
					mSampler::addToSetHoldIDs(nStance.stanceIndex, goal_stances);
				}
			}

			stance_nodes.push_back(nStance);
			if (stance_nodes.size() % 100 ==0)  //don't printf every node, will slow things down
				printf("Number of nodes: %d\n",stance_nodes.size());
			int index = stance_nodes.size() - 1;
			stance_id = index;
			stanceToNode[stanceToKey(nStance.hold_ids)]=index;
		}
		else
		{
			//if (stance_nodes[stance_id].stanceIndex == 1575)
			//{
			//	int notifyme= 1;
			//}
			//if (stance_nodes[stance_id].stanceIndex != stance_id)
			//{
			//	int notifyme = 1;
			//}
			if (stance_nodes[stance_id].stanceIndex == _fromIndex)
			{
				return stance_nodes[stance_id].stanceIndex;
			}
		}

		if (mSampler::addToSetHoldIDs(stance_nodes[_fromIndex].stanceIndex, stance_nodes[stance_id].parentStanceIds))
		{
			stance_nodes[stance_id].isItTried_to_father.push_back(false);
			if (stance_nodes[stance_id].parentStanceIds.size() > maxGraphDegree)
				maxGraphDegree = stance_nodes[stance_id].parentStanceIds.size();
		}

		if (mSampler::addToSetHoldIDs(stance_nodes[stance_id].stanceIndex, stance_nodes[_fromIndex].childStanceIds))
		{
			stance_nodes[_fromIndex].cost_moveLimb_to_child.push_back(getCostMovementLimbs(stance_nodes[_fromIndex].hold_ids, stance_nodes[stance_id].hold_ids));
			stance_nodes[_fromIndex].cost_transition_to_child.push_back(getCostTransition(stance_nodes[_fromIndex].hold_ids, stance_nodes[stance_id].hold_ids));
			stance_nodes[_fromIndex].isItTried_to_child.push_back(false);

			if (stance_nodes[_fromIndex].childStanceIds.size() > maxGraphDegree)
				maxGraphDegree = stance_nodes[_fromIndex].childStanceIds.size();
		}

		return stance_id;
	}

	int mRootGraph;

	std::vector<int> goal_stances;
	std::vector<mStanceNode> stance_nodes;
	unsigned int maxGraphDegree;
};

class mTrialStructure
{
public:
	mTrialStructure(int iTreeNode, int iGraPhNode, int iGraphFatherNodeId, float iValue)
	{
		_treeNodeId = iTreeNode;
		_graphNodeId = iGraPhNode;
		_graphFatherNodeId = iGraphFatherNodeId;
		val = iValue;
	}

	int _treeNodeId;
	int _graphNodeId;
	int _graphFatherNodeId;
	float val;
};

class mRRT
{
public:
	mSampler mClimberSampler;

	mStanceGraph mySamplingGraph;

	std::vector<mNode> mRRTNodes; // each node has the state of the agent
	KDTree<int> mKDTreeNodeIndices; // create an index on mRRTNodes
	int indexRootTree;

	robot_CPBP* mController;
	SimulationContext* mContextRRT;
	mSampleStructure mSample;
	
	int itr_optimization_for_each_sample;
	int max_itr_optimization_for_each_sample;
	int max_waste_interations;
	int max_no_improvement_iterations;

	int total_samples_num;
	int accepted_samples_num;

	// variable for playing animation
	BipedState lOptimizationBipedState;
	std::vector<int> path_nodes_indices;
	bool isPathFound;
	int lastPlayingNode;
	int lastPlayingStateInNode;
	bool isNodeShown;
	double cTimeElapsed;
	bool isAnimationFinished;

	// handling initial stance
	std::vector<int> initial_stance;
	bool initialstance_exists_in_tree;
	int index_node_initial_stance;

	// hanging leg when they are on a same hold
	bool isReachedLegs[2];

	// growing tree given a path by A*
	std::vector<mTrialStructure> mPriorityQueue;
	std::vector<std::vector<int>> mNodeIDsForPath;
	std::vector<bool> isItReachedFromToOnPath;
	std::vector<std::vector<int>> desiredPath;
	std::vector<int> mTriedPathIndicies;
	float cCostAStarPath;

	std::vector<int> lStancePath;
	float lastTimeNewPathReturned;

	//handling offline - online method
	enum targetMethod{Offiline = 0, Online = 1};

	// handling reached goals
	std::vector<int> goal_nodes;
	int cGoalPathIndex;
	bool isGoalReached;
	float cCostGoalMoveLimb;
	float cCostGoalControlCost;

	// handling rewiring
	int numRefine; // for debug
	int numPossRefine; // for debug
	std::vector<Vector2> mTransitionFromTo;

	// variables for sampling
	std::vector<int> m_reached_hold_ids; // reached to holds IDs
	std::vector<int> m_future_reachable_hold_ids; // future reachable holds are used to sample hand pos based on the calculated gain

	void resetTree()
	{
		mRRTNodes.clear();
		mKDTreeNodeIndices = KDTree<int>((int)(mController->startState.bodyStates.size() * 3));

		mSample = mSampleStructure();

		path_nodes_indices.clear();

		// growing tree given a path by A*
		mPriorityQueue.clear();
		mNodeIDsForPath.clear();
		isItReachedFromToOnPath.clear();
		desiredPath.clear();
		mTriedPathIndicies.clear();

		lStancePath.clear();

		goal_nodes.clear();

		mTransitionFromTo.clear();

		m_reached_hold_ids.clear();
		m_future_reachable_hold_ids.clear();

		initialstance_exists_in_tree = false;
		index_node_initial_stance = -1;

		mySamplingGraph.reset();
		mySamplingGraph.mClimberSampler = &mClimberSampler;
		mySamplingGraph.goalPos = mContextRRT->getGoalPos();
		mySamplingGraph.buildGraph(mController->startState.hold_bodies_ids, initial_stance);

		// handling reached goal
		isGoalReached = false;
		cCostGoalControlCost = -1;
		cCostGoalMoveLimb = -1;
		cGoalPathIndex = 0;
		numRefine = 0;
		numPossRefine = 0;

		isReachedLegs[0] = false;
		isReachedLegs[1] = false;
		UpdateTree(mController->startState, -1, std::vector<BipedState>(), mSampleStructure()); // -1 means root
		indexRootTree = 0;

		max_waste_interations = useOfflinePlanning ? 0 : 5;//5 if online
		int tmp_max_num_chosen_iterations =  6 * int(cTime / nPhysicsPerStep) + 30;
		//the online method is not working good with 3*(int)(cTime/nPhysicsPerStep), its value should be around 50
		max_itr_optimization_for_each_sample = useOfflinePlanning ? 3*(int)(cTime/nPhysicsPerStep) : (int)(1.5f * cTime);
		itr_optimization_for_each_sample = 0;
		//the online method is not working good with (int)(cTime/nPhysicsPerStep)/4, its value should be around 8
		max_no_improvement_iterations = useOfflinePlanning ? 5 : (int)(cTime/4);

		total_samples_num = 0;
		accepted_samples_num = 0;

		// variable for playing animation
		lOptimizationBipedState = mController->startState.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID());
		isPathFound = false;
		lastPlayingNode = 0;
		lastPlayingStateInNode = 0;
		isNodeShown = false;
		cTimeElapsed = 0.0f;
		isAnimationFinished = false;

		cCostAStarPath = 0.0f;

		lastTimeNewPathReturned = 0;
	}

	mRRT(SimulationContext* iContextRRT, robot_CPBP* iController, int _nCol)
		:mClimberSampler(iContextRRT), mKDTreeNodeIndices((int)(iController->startState.bodyStates.size() * 3))
	{
		mController = iController;
		mContextRRT = iContextRRT;

		//if (_nCol >= 2)
		//{
		_nCol = 2;
		// handling initial stance
		initial_stance.push_back(-1);
		initial_stance.push_back(-1);
		initial_stance.push_back(2 + _nCol - 2);
		initial_stance.push_back(3 + _nCol - 2);
		//}
		//else
		//{
			// handling initial stance
		//	initial_stance.push_back(-1);
		//	initial_stance.push_back(-1);
		//	initial_stance.push_back(1);
		//	initial_stance.push_back(1);
		//}
		initialstance_exists_in_tree = false;
		index_node_initial_stance = -1;

		mySamplingGraph.mClimberSampler = &mClimberSampler;
		mySamplingGraph.goalPos = mContextRRT->getGoalPos();
		mySamplingGraph.buildGraph(mController->startState.hold_bodies_ids, initial_stance);

		// handling reached goal
		isGoalReached = false;
		cCostGoalControlCost = -1;
		cCostGoalMoveLimb = -1;
		cGoalPathIndex = 0;
		numRefine = 0;
		numPossRefine = 0;

		isReachedLegs[0] = false;
		isReachedLegs[1] = false;
		UpdateTree(mController->startState, -1, std::vector<BipedState>(), mSampleStructure()); // -1 means root
		indexRootTree = 0;

		max_waste_interations = useOfflinePlanning ? 0 : 5;//5 if online
		int tmp_max_num_chosen_iterations =  6 * int(cTime / nPhysicsPerStep) + 30;
		//the online method is not working good with 3*(int)(cTime/nPhysicsPerStep), its value should be around 50
		max_itr_optimization_for_each_sample = useOfflinePlanning ? 3*(int)(cTime/nPhysicsPerStep) : (int)(1.5f * cTime);
		itr_optimization_for_each_sample = 0;
		//the online method is not working good with (int)(cTime/nPhysicsPerStep)/4, its value should be around 8
		max_no_improvement_iterations = useOfflinePlanning ? 5 : (int)(cTime/4);

		total_samples_num = 0;
		accepted_samples_num = 0;

		// variable for playing animation
		lOptimizationBipedState = mController->startState.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID());
		isPathFound = false;
		lastPlayingNode = 0;
		lastPlayingStateInNode = 0;
		isNodeShown = false;
		cTimeElapsed = 0.0f;
		isAnimationFinished = false;

		cCostAStarPath = 0.0f;

		lastTimeNewPathReturned = 0;
	} 

	void mRunPathPlanner(targetMethod itargetMethd, bool advance_time, bool flag_play_animation, mCaseStudy icurrent_case_study, float _time)
	{
		if (itargetMethd == targetMethod::Offiline)
		{
			mRunPathPlannerOffline(flag_play_animation, advance_time, icurrent_case_study, _time);
		}
		else
		{
			mRunPathPlannerOnline(advance_time, flag_play_animation, icurrent_case_study, _time);
		}
	}

	void mPrintStuff()
	{
		rcPrintString("Cost of A* prune stance path: %f \n",cCostAStarPath);
		//compute destination stance cost just to print out the components
		//mySamplingGraph.getCostAtNode(mSample.desired_hold_ids,true);

		/*printf("\n current: %d, %d, %d, %d, start = %d, %d, %d, %d, desreid= %d, %d, %d, %d, TO: %d,%d,%d,%d \n", mController->startState.hold_bodies_ids[0]
							, mController->startState.hold_bodies_ids[1], mController->startState.hold_bodies_ids[2], mController->startState.hold_bodies_ids[3]
							,mSample.initial_hold_ids[0], mSample.initial_hold_ids[1], mSample.initial_hold_ids[2], mSample.initial_hold_ids[3]
							,mSample.desired_hold_ids[0], mSample.desired_hold_ids[1], mSample.desired_hold_ids[2], mSample.desired_hold_ids[3]
							,mSample.to_hold_ids[0], mSample.to_hold_ids[1], mSample.to_hold_ids[2], mSample.to_hold_ids[3]);

		if (mSample.toNode != -1)
		{
			printf("\n //////////////  Error: %f ////////////////// \n", mRRTNodes[mSample.toNode].getDisFrom(mController->startState));
		}
		else
		{
			printf("\n //////////////  Error: %f ////////////////// \n", -1.0f);
		}

		printf("\n num nodes: %d,from node: %d, cCostGoal:%f, NRef:%d, NPRef:%d \n", mRRTNodes.size(), mSample.closest_node_index, cCostGoal, numRefine, numPossRefine);*/
	}

	void drawTree()
	{
		float delta_x = 2.5f;

		for (unsigned int n = 0; n < mRRTNodes.size(); n++)
		{
			std::vector<Vector3> cNode_hold_points;
			float cNodeHoldsSize = 0;
			 mNode& cNode = mRRTNodes[n];

			 mClimberSampler.getHoldStancePosFrom(cNode.cNodeInfo.hold_bodies_ids, cNode_hold_points, cNodeHoldsSize);
			 Vector3 ctrunkPos = cNode.cNodeInfo.computeCOM(); //mClimberSampler.getHoldStancePosFrom(cNode.cNodeInfo.hold_bodies_ids, cNode_hold_points, cNodeHoldsSize); //cNode.cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);

			 if (cNode.mChildrenIndices.size() > 0)
			 {
				 for (unsigned int c = 0; c < cNode.mChildrenIndices.size(); c++)
				 {
					// std::vector<Vector3> ccNode_hold_points;
					// float ccNodeHoldsSize = 0;
					 mNode& ccNode = mRRTNodes[cNode.mChildrenIndices[c]];
					 Vector3 cctrunkPos = ccNode.cNodeInfo.computeCOM();//mClimberSampler.getHoldStancePosFrom(ccNode.cNodeInfo.hold_bodies_ids, ccNode_hold_points, ccNodeHoldsSize);

					 float p1[] = {ctrunkPos.x() + delta_x,ctrunkPos.y()-0.1f,ctrunkPos.z()};
					 float p2[] = {cctrunkPos.x() + delta_x,cctrunkPos.y()-0.1f,cctrunkPos.z()};

					 Vector3 _color(1.0f,1.0f,1.0f); // draw line white for old goals or already explored tree
					 if (ccNode._goalNumNodeAddedFor == goal_nodes.size())
					 {
						 _color = Vector3(1.0f,0.0f,1.0f); // draw line magenta if it is added for current goal
					 }
					 
					 rcSetColor(_color.x(),_color.y(),_color.z());
					 rcDrawLine(p1, p2);
				 }
			 }
			 else
			 {
				 if (mSample.closest_node_index != cNode.nodeIndex)
				 {
					 Vector3 _color(1.0f,0.0f,0.0f);
					 if (cNode.cNodeInfo.hold_bodies_ids[2] != -1 && (cNode_hold_points[2] - mContextRRT->getGoalPos()).norm() < 0.1f)
					 {
						 _color = Vector3(0.0f,1.0f,0.0f);
					 }
					 if (cNode.cNodeInfo.hold_bodies_ids[3] != -1 && (cNode_hold_points[3] - mContextRRT->getGoalPos()).norm() < 0.1f)
					 {
						 _color = Vector3(0.0f,1.0f,0.0f);
					 }
					 robot_CPBP::drawCross(ctrunkPos + Vector3(0.0f + delta_x,-0.1f,0.0f), _color);
				 }
			 }
			 if (mSample.closest_node_index == cNode.nodeIndex)
			 {
				 Vector3 cctrunkPos = mController->bestCOMPos;
				 float p1[] = {ctrunkPos.x() + delta_x,ctrunkPos.y()-0.1f,ctrunkPos.z()};
				 float p2[] = {cctrunkPos.x() + delta_x,cctrunkPos.y()-0.1f,cctrunkPos.z()};

				 Vector3 _color(1.0f,0.0f,1.0f); // draw line magenta if it is added for current goal

				 rcSetColor(_color.x(),_color.y(),_color.z());
				 rcDrawLine(p1, p2);
				 robot_CPBP::drawCross(cctrunkPos + Vector3(0.0f + delta_x,-0.1f,0.0f), _color);
			 }
		}

		return;
	}

private:
	/////////////////////////////////////////////// choose offline or online mode
	void mRunPathPlannerOffline(bool flag_play_animation, bool advance_time, mCaseStudy icurrent_case_study, float _time)
	{
		if (!flag_play_animation)
		{
			// reverting to last state before animation playing
			if (mSample.restartSampleStartState && mSample.isSet)
			{
				//mController->startState = lOptimizationBipedState; // this is for online mode
				mController->startState = mRRTNodes[mSample.closest_node_index].cNodeInfo;
			}

			// get another sigma to simulate forward
			if (!mSample.isSet || mSample.isReached || mSample.isRejected || itr_optimization_for_each_sample < max_waste_interations)
			{
				if (!mSample.isSet || mSample.isReached || mSample.isRejected)
					mSample = mRRTTowardSamplePath2(mSample, _time); 
				mSample.statesFromTo.clear();

				//initiate before simulation
				if (mSample.closest_node_index != -1)
				{
					mController->startState = mRRTNodes[mSample.closest_node_index].cNodeInfo;
				}
				// means to restart to initial setting for holds
				if (mSample.isSet)
				{
					setArmsLegsHoldPoses();
				}
			}

			if (mSample.isSet)
			{
//				rcPrintString("Low-level controller iteration %d",itr_optimization_for_each_sample);
				// when it is false, it mean offline optimization
				mSteerFunc(false);
				if (advance_time)
					itr_optimization_for_each_sample++;

				// offline optimization is done, simulate forward on the best trajectory
				if ((advance_time && mSample.numItrFixedCost > max_no_improvement_iterations)
					|| itr_optimization_for_each_sample > max_itr_optimization_for_each_sample)
				{
					itr_optimization_for_each_sample = 0;
					bool flag_simulation_break = false;
					int maxTimeSteps=nTimeSteps/nPhysicsPerStep;
					if (optimizerType==otCMAES)
					{
						maxTimeSteps=mController->bestCmaesTrajectory.nSteps;
						/*if (nPhysicsPerStep!=1)
							Debug::throwError("CMAES does not support other than nPhysicsPerStep==1");*/
					}
					for (int cTimeStep = 0; cTimeStep < maxTimeSteps && !flag_simulation_break; cTimeStep++)
					{
						mStepOptimization(cTimeStep);
						// connect contact pos i to the desired hold pos i if some condition (for now just distance) is met
						m_Connect_Disconnect_ContactPoint(mSample.desired_hold_ids);

						// if the simulation is rejected a new sample should be generated
						// if the simulation is accepted the control sequence should be restored and then new sample should be generated
						mAcceptOrRejectSample(mSample, targetMethod::Offiline);

						if (mSample.isRejected)
							flag_simulation_break = true;
					}
					if (!flag_simulation_break)
					{
						// we just set the termination condition true to add new node in offline
						itr_optimization_for_each_sample = max_itr_optimization_for_each_sample + 1;

						mAcceptOrRejectSample(mSample, targetMethod::Offiline);
						mSample.isReached = true;
					}
				}

				mPrintStuff();
			}
			else
			{
				mController->takeStep();
			}
			if (mSample.closest_node_index >= 0)
				mSample.draw_ws_points(mRRTNodes[mSample.closest_node_index].cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk));
			
//			// save last optimization state
//			lOptimizationBipedState = mController->startState.getNewCopy(lOptimizationBipedState.saving_slot_state, mContextRRT->getMasterContextID());//
			isPathFound = false;
		}
		else
		{
			playAnimation(icurrent_case_study);
		}
		return;
	}

	void mRunPathPlannerOnline(bool advance_time, bool flag_play_animation, mCaseStudy icurrent_case_study, float _time)
	{
		if (!flag_play_animation)
		{
			if (mSample.restartSampleStartState && mSample.isSet)
			{
				mController->startState = lOptimizationBipedState;
			}
			if (!mSample.isSet || mSample.isReached || mSample.isRejected || itr_optimization_for_each_sample < max_waste_interations)
			{
				if (!mSample.isSet || mSample.isReached || mSample.isRejected)
					mSample = mRRTTowardSamplePath2(mSample, _time); //mRRTTowardSample(); 
				mSample.statesFromTo.clear();
				if (mSample.closest_node_index != -1)
				{
					mController->startState = mRRTNodes[mSample.closest_node_index].cNodeInfo;
				}
				// means to restart to initial setting for holds
				if (mSample.isSet)
				{
					setArmsLegsHoldPoses();
				}
			}

			if (mSample.isSet)
			{
				mSteerFunc(advance_time);

				if (advance_time)
				{
					// connect contact pos i to the desired hold pos i if some condition (for now just distance) is met
					m_Connect_Disconnect_ContactPoint(mSample.desired_hold_ids);

					// if the simulation is rejected a new sample should be generated
					// if the simulation is accepted the control sequence should be restored and then new sample should be generated
					mAcceptOrRejectSample(mSample, targetMethod::Online);

					itr_optimization_for_each_sample++;
				}
				mPrintStuff();
			}
			else
			{
				mController->takeStep();
			}
			if (mSample.closest_node_index >= 0)
				mSample.draw_ws_points(mRRTNodes[mSample.closest_node_index].cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk));
			
			// save last optimization state
			lOptimizationBipedState = mController->startState.getNewCopy(lOptimizationBipedState.saving_slot_state, mContextRRT->getMasterContextID());
			isPathFound = false;
		}
		else
		{
			playAnimation(icurrent_case_study);
		}
		return;
	}

	/////////////////////////////// my method /////////////////////////////////////////////////////////////
	mSampleStructure mRRTTowardSamplePath2(mSampleStructure& pSample, float _time)
	{		
		updatePriorityQueueReturnSample2(pSample);

		if (mPriorityQueue.size() == 0)
		{
			updateGraphTransitionValues2(_time);

			std::vector<int> nStancePath = mySamplingGraph.solveGraph(cCostAStarPath, mStanceGraph::mAlgSolveGraph::myDijkstraHueristic);

			if (!mNode::isSetAEqualsSetB(nStancePath, lStancePath))
			{
				lStancePath = nStancePath;
				lastTimeNewPathReturned = _time;
			}

			desiredPath = mySamplingGraph.returnPath();

			if (desiredPath.size() == 0)
			{
				maxNumSimulatedPaths = (int)goal_nodes.size();
			}

			getAllNodesAroundThePath2(desiredPath);

			mTriedPathIndicies.clear();
		}

		if (mPriorityQueue.size() > 0)
		{
			mTrialStructure _tryFromTo = mPriorityQueue[0];
			mPriorityQueue.erase(mPriorityQueue.begin());

			return retSampleFromGraph(_tryFromTo._graphFatherNodeId, _tryFromTo._graphNodeId);
		}
		return mSampleStructure();
	}

	void updatePriorityQueueReturnSample2(mSampleStructure& pSample)
	{
		if (pSample.isSet)
		{
			if (isItReached(pSample))
			{
				int index_on_path = mTriedPathIndicies[mTriedPathIndicies.size() - 1];
				isItReachedFromToOnPath[index_on_path - 1] = true;
				std::vector<int> mNodeIDsForPath_i;
				mNodeIDsForPath_i.push_back(pSample.toNode);
					//getAllTreeNodesForPathIndex2(desiredPath, index_on_path);

				if (mNodeIDsForPath_i.size() > 0)
				{
					mNodeIDsForPath[index_on_path] = mNodeIDsForPath_i;
					if (index_on_path < (int)isItReachedFromToOnPath.size())
					{
						if (!mSampler::isInSetHoldIDs(index_on_path + 1, mTriedPathIndicies)) //!isItReachedFromToOnPath[index_on_path] && 
						{
							mTrialStructure nT(-1, index_on_path+1, index_on_path, -1);
							if (!isTrialStrExistsInQueue(nT))
								mPriorityQueue.insert(mPriorityQueue.begin(), nT);
						}
					}
				}
			}
			else
			{
				int index_on_path = mTriedPathIndicies[mTriedPathIndicies.size() - 1];
				isItReachedFromToOnPath[index_on_path - 1] = false;
			}
		}

		if (mPriorityQueue.size() == 0 && mTriedPathIndicies.size() == 0)
		{
			for (unsigned int i = 0; i < isItReachedFromToOnPath.size(); i++)
			{
				if (!isItReachedFromToOnPath[i] && mNodeIDsForPath[i].size() > 0)
				{
					mTrialStructure nT(-1, i+1, i, -1);
					if (!isTrialStrExistsInQueue(nT))
						mPriorityQueue.push_back(nT);
				}
			}
		}

		return;
	}

	void getAllNodesAroundThePath2(std::vector<std::vector<int>>& dPath)
	{
		mNodeIDsForPath.clear();
		isItReachedFromToOnPath.clear();

		std::vector<std::vector<int>> mNodeIDsForStancePath;

		for (int i = 0; i < (int)dPath.size() - 1; i++)
		{
			isItReachedFromToOnPath.push_back(false);
		}

		// nodes that are representing stance path
		for (int i = 0; i < (int)dPath.size(); i++)
		{
			std::vector<int> mNodeIDsForStancePath_i;
			std::vector<int> mNodeIDsForPath_i = getAllTreeNodesForPathIndex2(dPath, i, mNodeIDsForStancePath_i);

			mNodeIDsForStancePath.push_back(mNodeIDsForStancePath_i);
			mNodeIDsForPath.push_back(mNodeIDsForPath_i);
		}
		//Uncomment to get all the nodes instead of just the one that follows the stance path
		//for (int i = 0; i < (int)dPath.size(); i++)
		//{
		//	if (mNodeIDsForPath[i].size() == 0)
		//	{
		//		mNodeIDsForPath[i] = mNodeIDsForStancePath[i];
		//	}
		//}
		return;
	}

	std::vector<int> getAllTreeNodesForPathIndex2(std::vector<std::vector<int>>& dPath, int i
		, std::vector<int>& mNodeIDsForStancePath_i)
	{
		float cCost = 0.0f;
		std::vector<int> s_i; 
		std::vector<int> s_i_1;

		std::vector<int> mNodeIDsForPath_i;
		if (i == 0)
		{
			mNodeIDsForPath_i.push_back(0);
			return mNodeIDsForPath_i;
		}

		for (int j = 0; j < (int)dPath.size(); j++)
		{
			s_i = dPath[j];
			if (j-1 >= 0)
			{
				s_i_1 = dPath[j-1];
				cCost += mNode::getDiffBtwSetASetB(s_i_1, s_i);
			}
			if (j == i)
			{
				break;
			}
		}

		// now we find all nodes that represent stance i in dPath and they should have come from i - 1
		std::vector<int> s_i_p1;
		if (i+1 < (int)dPath.size())
		{
			s_i_p1 = dPath[i+1];
		}

		for (unsigned int n = 0; n < mRRTNodes.size(); n++)
		{
			mNode* tNode_n = &mRRTNodes[n];

			if (!tNode_n->isNodeEqualTo(s_i))
			{
				continue;
			}

			if (s_i_p1.size() > 0)
			{
				if (tNode_n->isInTriedHoldSet(s_i_p1) && !isStanceExistInTreeNodeChildren(tNode_n, s_i_p1))
				{
					continue;
				}
			}

			if (i - 1 >= 0)
			{
				if (mSampler::isInSetHoldIDs(tNode_n->mFatherIndex, mNodeIDsForPath[i-1]))
				{
					isItReachedFromToOnPath[i-1] = true;
					mNodeIDsForPath_i.push_back(n); // representing s_i in the path
				}
			}
			else if (tNode_n->mFatherIndex == -1)
			{
				mNodeIDsForPath_i.push_back(n);
			}
			if (!isStanceExistInTreeNodeChildren(tNode_n, s_i_p1))
			{
				mNodeIDsForStancePath_i.push_back(n);
			}
			//if (s_i_1.size() > 0)
			//{
			//	bool flag_add = false;
			//	/*if (getNodeCost(tNode_n) <= cCost)
			//	{
			//		flag_add = true;
			//	}*/
			//	if (tNode_n->mFatherIndex == -1)
			//	{
			//		if (!flag_add)
			//			continue;
			//	}
			//	mNode* tNode_f = &mRRTNodes[tNode_n->mFatherIndex];
			//	if (!tNode_f->isNodeEqualTo(s_i_1))
			//	{
			//		if (!flag_add)
			//			continue;
			//	}
			//	else
			//	{
			//		isItReachedFromToOnPath[i-1] = true;
			//	}
			//}
		}
		return mNodeIDsForPath_i;
	}

	void updateGraphTransitionValues2(float total_time_elasped)
	{
		float timeDiff = total_time_elasped;
		timeDiff = total_time_elasped - lastTimeNewPathReturned;

		bool flag_path_changed = false;
		for (unsigned int i = 0; i < mTriedPathIndicies.size(); i++)
		{
			int index_on_path = mTriedPathIndicies[i];
			if (!isItReachedFromToOnPath[index_on_path - 1])
			{
				bool is_changed_val = mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_on_path-1), mySamplingGraph.getIndexStanceNode(index_on_path));//, timeDiff < trialTimeBeforeChangePath);
				if (!flag_path_changed)
				{
					flag_path_changed = is_changed_val;
				}
			}
		}

		if (flag_path_changed)
		{
			mySamplingGraph.initializeOpenListAStarPrune();
		}

		if (!flag_path_changed && mySamplingGraph.retPath.size() > 0) // path is reached, A* should return another path
		{
			if (!mSampler::isInSampledStanceSet(mySamplingGraph.retPath, mySamplingGraph.m_found_paths))
			{
				mySamplingGraph.m_found_paths.push_back(mySamplingGraph.retPath);
			}
		}

		return;
	}

	/////////////////////////////// utilities for samplings form graph /////////////////////////////////////
	template<typename T>
	void createParticleList(unsigned int mN, float cWeight, T cSample,std::vector<float>& weights, float& sumWeights, std::vector<T>& samples)
	{
		unsigned int j = 0;
		for (j = 0; j < weights.size(); j++)
		{
			if (cWeight > weights[j])
			{
				break;
			}
		}
		if (weights.size() < mN)
		{
			sumWeights += cWeight;
			weights.insert(weights.begin() + j, cWeight);
			samples.insert(samples.begin() + j, cSample);
		}
		else
		{
			sumWeights += cWeight;
			weights.insert(weights.begin() + j, cWeight);
			samples.insert(samples.begin() + j, cSample);

			sumWeights -= weights[weights.size() - 1];
			weights.erase(weights.begin() + (weights.size() - 1));
			samples.erase(samples.begin() + (samples.size() - 1));
		}
	}

	bool isItReached(mSampleStructure& pSample)
	{
		if (mNode::isSetAEqualsSetB(pSample.desired_hold_ids, mController->startState.hold_bodies_ids))
		{
			return true;
		}
		if (pSample.desired_hold_ids[0] == pSample.desired_hold_ids[1] && pSample.desired_hold_ids[1] > -1)
		{
			if (pSample.desired_hold_ids[2] == mController->startState.hold_bodies_ids[2] && pSample.desired_hold_ids[3] == mController->startState.hold_bodies_ids[3])
			{
				if (isReachedLegs[0] && isReachedLegs[1])
				{
					return true;
				}
			}
		}
		return false;
	}

	void updateGraphTransitionValues()
	{
		bool flag_path_changed = false;
		for (unsigned int i = 0; i < mTriedPathIndicies.size(); i++)
		{
			int index_on_path = mTriedPathIndicies[i];
			if (!isItReachedFromToOnPath[index_on_path - 1])
			{
				bool is_changed_val = mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_on_path-1), mySamplingGraph.getIndexStanceNode(index_on_path));
				if (!flag_path_changed)
				{
					flag_path_changed = is_changed_val;
				}
			}
		}

		if (!flag_path_changed && mySamplingGraph.retPath.size() > 0) // path is reached, A* should return another path
		{
			if (!mSampler::isInSampledStanceSet(mySamplingGraph.retPath, mySamplingGraph.m_found_paths))
			{
				mySamplingGraph.m_found_paths.push_back(mySamplingGraph.retPath);
				mySamplingGraph.setCurPathTriedFather();
			}
		}

		/*for (int i = isItReachedFromToOnPath.size() - 1; i >= 0 && !flag_path_changed; i--)
		{
			int index_on_path = i + 1;
			if (isItReachedFromToOnPath[index_on_path - 1])
			{
				flag_path_changed = mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_on_path-1), mySamplingGraph.getIndexStanceNode(index_on_path));
			}
		}*/
		/*for (int i = mNodeIDsForPath.size() - 1; i >= 0 && !flag_path_changed; i--)
		{
			int index_on_path = i;
			if (mNodeIDsForPath[i].size() > 0)
			{
				flag_path_changed = mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_on_path-1), mySamplingGraph.getIndexStanceNode(index_on_path));
			}
		}*/

		return;
	}

	mSampleStructure mRRTTowardSamplePath(mSampleStructure& pSample)
	{		
		updatePriorityQueueReturnSample(pSample);

		if (mPriorityQueue.size() == 0)
		{
			updateGraphTransitionValues();

			float mVal;
			mySamplingGraph.solveGraph(mVal, mStanceGraph::mAlgSolveGraph::myDijkstraHueristic);

			desiredPath = mySamplingGraph.returnPath();

			getAllNodesAroundThePath(desiredPath);

			mTriedPathIndicies.clear();

			printf("\n solve, %f", mVal);
		}

		if (mPriorityQueue.size() > 0)
		{
			mTrialStructure _tryFromTo = mPriorityQueue[0];
			mPriorityQueue.erase(mPriorityQueue.begin());

			return retSampleFromGraph(_tryFromTo._graphFatherNodeId, _tryFromTo._graphNodeId);
		}
		return mSampleStructure();
	}

	bool isTrialStrExistsInQueue(mTrialStructure& nT)
	{
		for (unsigned int i = 0; i < mPriorityQueue.size(); i++)
		{
			mTrialStructure cT = mPriorityQueue[i];
			if (cT._graphNodeId == nT._graphNodeId && cT._graphFatherNodeId == nT._graphFatherNodeId)
			{
				return true;
			}
		}
		return false;
	}

	void updatePriorityQueueReturnSample(mSampleStructure& pSample)
	{
		if (pSample.isSet)
		{
			if (isItReached(pSample))
			{
				int index_on_path = mTriedPathIndicies[mTriedPathIndicies.size() - 1];
				isItReachedFromToOnPath[index_on_path - 1] = true;
				std::vector<int> mNodeIDsForPath_i = getAllTreeNodesForPathIndex(desiredPath, index_on_path);

				if (mNodeIDsForPath_i.size() > 0)
				{
					mNodeIDsForPath[index_on_path] = mNodeIDsForPath_i;
					if (index_on_path < (int)isItReachedFromToOnPath.size())
					{
						if (!isItReachedFromToOnPath[index_on_path] && !mSampler::isInSetHoldIDs(index_on_path + 1, mTriedPathIndicies))
						{
							mTrialStructure nT(-1, index_on_path+1, index_on_path, -1);
							if (!isTrialStrExistsInQueue(nT))
								mPriorityQueue.insert(mPriorityQueue.begin(), nT);
						}
					}
				}
			}
			else
			{
				int index_on_path = mTriedPathIndicies[mTriedPathIndicies.size() - 1];
				isItReachedFromToOnPath[index_on_path - 1] = false;
			}
		}

		if (mPriorityQueue.size() == 0 && mTriedPathIndicies.size() == 0)
		{
			for (int i = isItReachedFromToOnPath.size() - 1; i >= 0; i--)
			{
				if (!isItReachedFromToOnPath[i] && mNodeIDsForPath[i].size() > 0)
				{
					mTrialStructure nT(-1, i+1, i, -1);
					if (!isTrialStrExistsInQueue(nT))
						mPriorityQueue.push_back(nT);
				}
			}
		}

		return;
	}

	mSampleStructure retSampleFromGraph(int _fromPathNodeID, int _toPathNodeID)
	{
		std::vector<int> transition_nodes = mNodeIDsForPath[_fromPathNodeID];

		std::vector<mTrialStructure> samples;
		std::vector<float> weights;
		float sumWeights = 0;

		for (unsigned int i = 0; i < transition_nodes.size(); i++)
		{
			mNode* tNodei = &mRRTNodes[transition_nodes[i]];

			std::vector<Vector3> sample_des_points; float mSizeDes = 0;
			Vector3 midDesPoint = mClimberSampler.getHoldStancePosFrom(desiredPath[_toPathNodeID], sample_des_points, mSizeDes);

			float value = sqrt(tNodei->getSumDisEndPosTo(desiredPath[_toPathNodeID], sample_des_points));

			if (tNodei->isInTriedHoldSet(desiredPath[_toPathNodeID]))
			{
				value *= 1000.0f;
			}

			float f = 1 / value;

			createParticleList<mTrialStructure>(10, f, 
				mTrialStructure(tNodei->nodeIndex, mySamplingGraph.getIndexStanceNode(_toPathNodeID), mySamplingGraph.getIndexStanceNode(_toPathNodeID-1), value)
													, weights, sumWeights, samples);
		}

		int rIndex = chooseIndexFromParticles(weights, sumWeights);
		if (rIndex >= 0)
		{
			mSampler::addToSetHoldIDs(_toPathNodeID, mTriedPathIndicies);

			mSampleStructure nSample = mClimberSampler.getSampleFrom(&mRRTNodes[samples[rIndex]._treeNodeId], mySamplingGraph.getStanceGraph(samples[rIndex]._graphNodeId), false);

			isReachedLegs[0] = true;
			isReachedLegs[1] = true;
			if (nSample.initial_hold_ids[0] == -1 && nSample.desired_hold_ids[0] != -1)
			{
				isReachedLegs[0] = false;
			}
			if (nSample.initial_hold_ids[1] == -1 && nSample.desired_hold_ids[1] != -1)
			{
				isReachedLegs[1] = false;
			}

			nSample.toNodeGraph = samples[rIndex]._graphNodeId;
			nSample.fromNodeGraph = samples[rIndex]._graphFatherNodeId;

			printf("\n fNode:%d, tNode:%d \n", nSample.fromNodeGraph, nSample.toNodeGraph);
			return nSample;
		}
		
		return mSampleStructure();
	}

	std::vector<int> getAllTreeNodesForPathIndex(std::vector<std::vector<int>>& dPath, int i)
	{
		float cCost = 0.0f;
		std::vector<int> s_i; 
		std::vector<int> s_i_1;

		for (int j = 0; j < (int)dPath.size(); j++)
		{
			s_i = dPath[j];
			if (j-1 >= 0)
			{
				s_i_1 = dPath[j-1];
				cCost += mNode::getDiffBtwSetASetB(s_i_1, s_i);
			}
			if (j == i)
			{
				break;
			}
		}

		// now we find all nodes that represent stance i in dPath and they should have come from i - 1
		std::vector<int> s_i_p1;
		if (i+1 < (int)dPath.size())
		{
			s_i_p1 = dPath[i+1];
		}

		std::vector<int> mNodeIDsForPath_i;
		for (unsigned int n = 0; n < mRRTNodes.size(); n++)
		{
			mNode* tNode_n = &mRRTNodes[n];

			if (!tNode_n->isNodeEqualTo(s_i))
			{
				continue;
			}

			if (s_i_p1.size() > 0)
			{
				if (tNode_n->isInTriedHoldSet(s_i_p1) && !isStanceExistInTreeNodeChildren(tNode_n, s_i_p1))
				{
					continue;
				}
			}

			if (s_i_1.size() > 0)
			{
				bool flag_add = false;
				if (getNodeCost(tNode_n, mCaseStudy::movingLimbs) <= cCost)
				{
					flag_add = true;
				}

				if (tNode_n->mFatherIndex == -1)
				{
					if (!flag_add)
						continue;
				}

				mNode* tNode_f = &mRRTNodes[tNode_n->mFatherIndex];

				if (!tNode_f->isNodeEqualTo(s_i_1))
				{
					if (!flag_add)
						continue;
				}
				else
				{
					isItReachedFromToOnPath[i-1] = true;
				}
			}

			mNodeIDsForPath_i.push_back(n); // representing s_i in the path
		}
		return mNodeIDsForPath_i;
	}

	bool isStanceExistInTreeNodeChildren(mNode* tNode_n, std::vector<int>& s_i_p1)
	{
		for (unsigned int c = 0; c < tNode_n->mChildrenIndices.size(); c++)
		{
			mNode* cNode = &mRRTNodes[tNode_n->mChildrenIndices[c]];
			if (cNode->isNodeEqualTo(s_i_p1))
			{
				return true;
			}
		}
		return false;
	}

	void getAllNodesAroundThePath(std::vector<std::vector<int>>& dPath)
	{
		mNodeIDsForPath.clear();
		isItReachedFromToOnPath.clear();

		for (int i = 0; i < (int)dPath.size() - 1; i++)
		{
			isItReachedFromToOnPath.push_back(false);
		}

		// nodes that are representing stance path
		for (int i = 0; i < (int)dPath.size(); i++)
		{
			std::vector<int> mNodeIDsForPath_i = getAllTreeNodesForPathIndex(dPath, i);

			mNodeIDsForPath.push_back(mNodeIDsForPath_i);
		}

		return;
	}

//	mSampleStructure mRRTTowardSamplePath(mSampleStructure& pSample)
//	{
//		if (!isItReached(pSample) && pSample.isSet)
//		{
//			mySamplingGraph.updateGraphNN(pSample.fromNodeGraph, pSample.toNodeGraph);
//		}
//		if (!isItReached(pSample) || !pSample.isSet)
//		{
//			float mVal;
//			mySamplingGraph.solveAStar(mVal);
//			mTriedPathIndicies.clear();
//
//			printf("\n solve, %f", mVal);
//		}
//
//		std::vector<std::vector<int>> desiredPath = mySamplingGraph.returnPath();
//
//		return getAllNodesAroundThePath(desiredPath);
//	}
//
//	mSampleStructure getAllNodesAroundThePath(std::vector<std::vector<int>>& dPath)
//	{
//		int index_next_stance_on_path = -1; // we should grow tree toward this index
//
//		std::vector<std::vector<int>> mNodeIDsForPath; // nodes that are representing stance path
//		float cCost = 0.0f;
//		for (int i = 0; i < (int)dPath.size(); i++)
//		{
//			// now we find all nodes that represent stance i in dPath and they should have come from i - 1
//			std::vector<int> s_i = dPath[i];
//			std::vector<int> s_i_1;
//			if (i-1 >= 0)
//			{
//				s_i_1 = dPath[i-1];
//				cCost += mNode::getDiffBtwSetASetB(s_i_1, s_i);
//			}
//
//			std::vector<int> mNodeIDsForPath_i;
//			for (unsigned int n = 0; n < mRRTNodes.size(); n++)
//			{
//				mNode* tNode_n = &mRRTNodes[n];
//
//				if (!tNode_n->isNodeEqualTo(s_i))
//				{
//					continue;
//				}
//
//				if (s_i_1.size() > 0)
//				{
//					if (getNodeCost(tNode_n) > cCost)
//					{
//						if (tNode_n->mFatherIndex == -1)
//						{
//							continue;
//						}
//
//						mNode* tNode_f = &mRRTNodes[tNode_n->mFatherIndex];
//
//						if (!tNode_f->isNodeEqualTo(s_i_1))
//						{
//							continue;
//						}
//					}
//				}
//
//				mNodeIDsForPath_i.push_back(n); // representing s_i in the path
//			}
//			if (mNodeIDsForPath_i.size() > 0)
//			{
//				mNodeIDsForPath.push_back(mNodeIDsForPath_i);
//				index_next_stance_on_path = i + 1;
//			}
//		}
//
//		//index_next_stance_on_path = mNodeIDsForPath.size();
//		if (index_next_stance_on_path >= (int)dPath.size())
//		{
//			index_next_stance_on_path = dPath.size() - 1;
//			mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_next_stance_on_path-1), mySamplingGraph.getIndexStanceNode(index_next_stance_on_path));
//			return mSampleStructure();
//		}
//
//		if (mSampler::isInSetHoldIDs(index_next_stance_on_path, mTriedPathIndicies))
//		{
//			mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_next_stance_on_path-1), mySamplingGraph.getIndexStanceNode(index_next_stance_on_path));
//			return mSampleStructure();
//		}
//		else
//		{
//			mTriedPathIndicies.push_back(index_next_stance_on_path);
//		}
//
//		/*std::vector<int> transition_nodes;
//		std::vector<int> mNodeIDsForPath_i;
//		for (unsigned int j = 0; j < mNodeIDsForPath[mNodeIDsForPath.size() - 1].size(); j++)
//		{
//			mNode* tNodei = &mRRTNodes[mNodeIDsForPath[mNodeIDsForPath.size() - 1][j]];
//			if (tNodei->isInTriedHoldSet(dPath[index_next_stance_on_path]))
//			{
//				continue;
//			}
//			mNodeIDsForPath_i.push_back(mNodeIDsForPath[mNodeIDsForPath.size() - 1][j]);
//		}
//		if (mNodeIDsForPath_i.size() == 0)
//		{
//			mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_next_stance_on_path-1), mySamplingGraph.getIndexStanceNode(index_next_stance_on_path));
//		}
//		else
//		{
//			transition_nodes = mNodeIDsForPath_i;
//		}*/
//
//		std::vector<int> transition_nodes = mNodeIDsForPath[mNodeIDsForPath.size()-1];
//
//		std::vector<mTrialStructure> samples;
//		std::vector<float> weights;
//		float sumWeights = 0;
//
//		for (unsigned int i = 0; i < transition_nodes.size(); i++)
//		{
//			mNode* tNodei = &mRRTNodes[transition_nodes[i]];
//
//			std::vector<Vector3> sample_des_points; float mSizeDes = 0;
//			Vector3 midDesPoint = mClimberSampler.getHoldStancePosFrom(dPath[index_next_stance_on_path], sample_des_points, mSizeDes);
//
//			float value = sqrt(tNodei->getSumDisEndPosTo(dPath[index_next_stance_on_path], sample_des_points)) + squared(dPath.size() - index_next_stance_on_path);
//
//			if (tNodei->isInTriedHoldSet(dPath[index_next_stance_on_path]))
//			{
//				value *= 1000.0f;
//			}
//
//			float f = 1 / value;
//
//			createParticleList<mTrialStructure>(10, f, 
//				mTrialStructure(tNodei->nodeIndex, mySamplingGraph.getIndexStanceNode(index_next_stance_on_path), mySamplingGraph.getIndexStanceNode(index_next_stance_on_path-1), value)
//													, weights, sumWeights, samples);
//		}
//
////		for (unsigned int i = getInitFrom(); i < mRRTNodes.size(); i++)
////		{
////			mNode* tNodei = &mRRTNodes[i];
//////			for (unsigned int j = 0; j < dPath.size(); j++)
//////			{
////			if (tNodei->isInTriedHoldSet(dPath[j]))
////			{
////				continue;
////			}
////			if (!mClimberSampler.isFromStanceToStanceValid(tNodei->cNodeInfo.hold_bodies_ids, dPath[j], !initialstance_exists_in_tree))
////			{
////				continue;
////			}
////
////			std::vector<Vector3> sample_des_points; float mSizeDes = 0;
////			Vector3 midDesPoint = mClimberSampler.getHoldStancePosFrom(dPath[j], sample_des_points, mSizeDes);
////
////			float value = sqrt(tNodei->getSumDisEndPosTo(dPath[j], sample_des_points)) + squared(dPath.size() - j);
////
////			float f = 1 / value;
////
////			createParticleList<mTrialStructure>(10, f, 
////				mTrialStructure(i, mySamplingGraph.getIndexStanceNode(j), mySamplingGraph.findStanceFrom(tNodei->cNodeInfo.hold_bodies_ids), value)
////												, weights, sumWeights, samples);
//////			}
////		}
//
//		int rIndex = chooseIndexFromParticles(weights, sumWeights);
//		if (rIndex >= 0)
//		{
//			mSampleStructure nSample = mClimberSampler.getSampleFrom(&mRRTNodes[samples[rIndex]._treeNodeId], mySamplingGraph.getStanceGraph(samples[rIndex]._graphNodeId), false);
//
//			isReachedLegs[0] = true;
//			isReachedLegs[1] = true;
//			if (nSample.initial_hold_ids[0] == -1 && nSample.desired_hold_ids[0] != -1)
//			{
//				isReachedLegs[0] = false;
//			}
//			if (nSample.initial_hold_ids[1] == -1 && nSample.desired_hold_ids[1] != -1)
//			{
//				isReachedLegs[1] = false;
//			}
//
//			nSample.toNodeGraph = samples[rIndex]._graphNodeId;
//			nSample.fromNodeGraph = samples[rIndex]._graphFatherNodeId;
//			return nSample;
//		}
//		else
//		{
//			mySamplingGraph.updateGraphNN(mySamplingGraph.getIndexStanceNode(index_next_stance_on_path-1), mySamplingGraph.getIndexStanceNode(index_next_stance_on_path));
//		}
//		return mSampleStructure();
//	}

	//////////////////////////////////// utilities for samplings ///////////////////////////////////////////

	void createParticleList(unsigned int mN, float cWeight, int cIndex,std::vector<float>& weights, float& sumWeights, std::vector<int>& indices)
	{
		unsigned int j = 0;
		for (j = 0; j < weights.size(); j++)
		{
			if (cWeight > weights[j])
			{
				break;
			}
		}
		if (weights.size() < mN)
		{
			sumWeights += cWeight;
			weights.insert(weights.begin() + j, cWeight);
			indices.insert(indices.begin() + j, cIndex);
		}
		else
		{
			sumWeights += cWeight;
			weights.insert(weights.begin() + j, cWeight);
			indices.insert(indices.begin() + j, cIndex);

			sumWeights -= weights[weights.size() - 1];
			weights.erase(weights.begin() + (weights.size() - 1));
			indices.erase(indices.begin() + (indices.size() - 1));
		}
	}

	void createParticleList(unsigned int mN, float& sumWeights, float cWeight, int cIndex, std::vector<int>& cSample, 
		std::vector<float>& weights, std::vector<int>& indices, std::vector<std::vector<int>>& samples)
	{
		unsigned int j = 0;
		for (j = 0; j < weights.size(); j++)
		{
			if (cWeight > weights[j])
			{
				break;
			}
		}
		if (weights.size() < mN)
		{
			sumWeights += cWeight;
			weights.insert(weights.begin() + j, cWeight);
			indices.insert(indices.begin() + j, cIndex);
			samples.insert(samples.begin() + j, cSample);
		}
		else
		{
			sumWeights += cWeight;
			weights.insert(weights.begin() + j, cWeight);
			indices.insert(indices.begin() + j, cIndex);
			samples.insert(samples.begin() + j, cSample);

			sumWeights -= weights[weights.size() - 1];
			weights.erase(weights.begin() + (weights.size() - 1));
			indices.erase(indices.begin() + (indices.size() - 1));
			samples.erase(samples.begin() + (samples.size() - 1));
		}
	}

	int chooseIndexFromParticles(std::vector<float>& weights, float sumWeights)
	{
		float preWeight = 0.0f;
		float p_r = mSampler::getRandomBetween_01();
		int cIndex = -1;
		for (unsigned int i = 0; i < weights.size(); i++)
		{
			float cWeight = (weights[i] / sumWeights) + preWeight;
			if (p_r > preWeight && p_r <= cWeight)
			{
				cIndex = i;
				break;
			}
			preWeight = cWeight;
		}
		
		return cIndex;
	}

	int numPossibleSolutionsForHandHold(int k)
	{
		int ret_ans = 1;
		for (int i = 0; i < 3; i++)
		{
			ret_ans *= k;
		}
		return 2 * ret_ans;
	}

	std::vector<int> sampleDesiredHoldsCenteringHand(Vector3& dPoint, SimulationContext::BodyName dSourceP)
	{
		int closest_hold_id = mClimberSampler.getClosestPoint(dPoint);

		std::vector<int> chosen_holds_ids = mClimberSampler.indices_lower_than[closest_hold_id];
		
		int id_hand = (int)(dSourceP - SimulationContext::BodyName::BodyLeftLeg);

		bool flag_found_hand_id = false;
		std::vector<int> leg_holds_samples;
		std::vector<int> hand_holds_samples;
		for (unsigned int i = getInitFrom(); i < mRRTNodes.size() && !flag_found_hand_id; i++)
		{
			mNode node_i = mRRTNodes[i];
			if (node_i.cNodeInfo.hold_bodies_ids[2] ==  closest_hold_id || node_i.cNodeInfo.hold_bodies_ids[3] ==  closest_hold_id)
			{
				flag_found_hand_id = true;
			}

			for (int j = 0 ; j < 4; j++)
			{
				if (j != id_hand)
				{
					if (mSampler::isInSetHoldIDs(node_i.cNodeInfo.hold_bodies_ids[j], chosen_holds_ids))
					{
						if (j == 0 || j == 1)
						{
							leg_holds_samples.push_back(node_i.cNodeInfo.hold_bodies_ids[j]);
						}
						else
						{
							hand_holds_samples.push_back(node_i.cNodeInfo.hold_bodies_ids[j]);
						}
					}
				}
			}
		}

		if (flag_found_hand_id || leg_holds_samples.size() == 0 || hand_holds_samples.size() == 0)
		{
			if (flag_found_hand_id)
			{
				leg_holds_samples.clear();
				hand_holds_samples.clear();
			}

			bool flag_add_hand = false;
			bool flag_add_leg = false;

			if (leg_holds_samples.size() == 0)
			{
				flag_add_leg = true;
			}
			if (hand_holds_samples.size() == 0)
			{
				flag_add_hand = true;
			}

			float midX = 0.0;
			float minZ = FLT_MAX;
			float maxZ = -FLT_MAX;
			for (unsigned int i = 0; i < chosen_holds_ids.size(); i++)
			{
				Vector3 h_i = mClimberSampler.getHoldPos(chosen_holds_ids[i]);
				midX += (1.0f / (float)(chosen_holds_ids.size())) * h_i.x();
				if (h_i.z() > maxZ)
				{
					maxZ = h_i.z();
				}
				if (h_i.z() < minZ)
				{
					minZ = h_i.z();
				}
			}

			float r = mSampler::getRandomBetween_01();
			Vector3 midPoint(midX, 0.0f, r * (maxZ - minZ) + minZ);

			for (unsigned int i = 0; i < chosen_holds_ids.size(); i++)
			{
				Vector3 hold_i_pos = mClimberSampler.getHoldPos(chosen_holds_ids[i]);
				float m_angle = SimulationContext::getAngleBtwVectorsXZ(Vector3(1.0f, 0.0f, 0.0f), hold_i_pos - midPoint);

				float cDis = (hold_i_pos - midPoint).norm();
				if (cDis < 0.01f)
				{
					if (flag_add_leg)
						leg_holds_samples.push_back(chosen_holds_ids[i]);
					if (flag_add_hand)
						hand_holds_samples.push_back(chosen_holds_ids[i]);
					continue;
				}

				if ((m_angle <= 0.3f * PI) && (m_angle >= -PI) || (m_angle >= 0.3f * PI)) 
				{
					if (flag_add_leg)
						leg_holds_samples.push_back(chosen_holds_ids[i]);
				}
				if ((m_angle >= -0.3f * PI) && (m_angle <= PI) || (m_angle >= -0.3f * PI)) 
				{
					if (flag_add_hand)
						hand_holds_samples.push_back(chosen_holds_ids[i]);
				}
			}
		}

		std::vector<int> desired_hold_ids; // left-leg, right-leg, left-hand, right-hand

		for (int i = 3; i >= 0; i--)
		{
			SimulationContext::BodyName cSourceP = (SimulationContext::BodyName)(SimulationContext::BodyName::BodyLeftLeg + i);
			if (cSourceP == dSourceP)
			{
				desired_hold_ids.insert(desired_hold_ids.begin(), closest_hold_id);
			}
			else
			{
				if (i <= 1)
				{
					int rIndex = mSampler::getRandomIndex(leg_holds_samples.size());
					desired_hold_ids.insert(desired_hold_ids.begin(), leg_holds_samples[rIndex]);
				}
				else
				{
					int rIndex = mSampler::getRandomIndex(hand_holds_samples.size());
					desired_hold_ids.insert(desired_hold_ids.begin(), hand_holds_samples[rIndex]);
				}
			}
		}
		return desired_hold_ids;
	}

	Vector3 getRandomHandPose(std::vector<int>& cReachableHoldIndices, std::vector<int>& cReachedHoldIndices, bool& flag_greedy_search)
	{
		// calculate gain for current future set of reachable holds
		// gain represents if this hold is set as hand pos, how many of other holds have the potential of being reached
		float p_r = mSampler::getRandomBetween_01();
		Vector3 goal_point = mContextRRT->getGoalPos();
		std::vector<int> choose_from_indices;

		std::vector<float> values_greedy;
		if (p_r < 0.7f && !isGoalReached)
		{
			for (unsigned int i = 0; i < cReachableHoldIndices.size(); i++)
			{
				int k = cReachableHoldIndices[i];
				float v_i = 0.0;
				for (unsigned int j = 0; j < mClimberSampler.indices_lower_than[k].size(); j++)
				{
					int c_index = mClimberSampler.indices_lower_than[k][j];
					if (!mSampler::isInSetHoldIDs(c_index, cReachedHoldIndices))
					{
						if (!mSampler::isInSetHoldIDs(k, choose_from_indices))
						{
							choose_from_indices.push_back(k);
						}
						v_i += 1 / (1 + (mClimberSampler.getHoldPos(c_index) - goal_point).norm());
					}
				}
				values_greedy.push_back(v_i);
			}
			flag_greedy_search = true;
		}
		else
		{
			flag_greedy_search = false;
		}

		if (!flag_greedy_search || choose_from_indices.size() == 0)
		{
			// random exploration helps us to cover the whole work space and fill the gaps, and eventually enables us to rewire
			for (unsigned int i = 0; i < cReachableHoldIndices.size(); i++)
			{
				int k = cReachableHoldIndices[i];
				if (mClimberSampler.indices_lower_than[k].size() > 1)
				{
					if (!mSampler::isInSetHoldIDs(k, choose_from_indices))
					{
						choose_from_indices.push_back(k);
					}
				}
			}
			for (unsigned int i = 0; i < cReachedHoldIndices.size(); i++)
			{
				int k = cReachedHoldIndices[i];
				if (mClimberSampler.indices_lower_than[k].size() > 1)
				{
					if (!mSampler::isInSetHoldIDs(k, choose_from_indices))
					{
						choose_from_indices.push_back(k);
					}
				}
			}
			flag_greedy_search = false;
		}

		std::vector<float> value_indices;
		float sum_values = 0.0f;
		for (unsigned int i = 0; i < choose_from_indices.size(); i++)
		{
			float v_i = 0.0f;

			if (flag_greedy_search)
			{
				v_i = values_greedy[i];
			}
			else
			{
				v_i = (float)(numPossibleSolutionsForHandHold(mClimberSampler.indices_lower_than[choose_from_indices[i]].size()));
			}

			if (v_i < 0)
			{
				v_i = 0.01f;
			}
			value_indices.push_back(v_i);
			sum_values += v_i;
		}
		int cIndex = chooseIndexFromParticles(value_indices, sum_values);

		if (cIndex >= 0)
			return mClimberSampler.getHoldPos(choose_from_indices[cIndex]);
		return mClimberSampler.getHoldPos(2);//mSampler::getRandomIndex(mClimberSampler.myHoldPoints.size()));
	}

	int getInitFrom()
	{
		int init_from = 0;
		if (initialstance_exists_in_tree)
		{
			init_from = index_node_initial_stance;
			if (init_from < 0)
				init_from = 0;
		}
		return init_from;
	}

	mSampleStructure mRRTTowardSample(mCaseStudy icurrent_case_study = mCaseStudy::movingLimbs)
	{
		bool flag_greedy_search = true;
		
		printf("\n 1");

		//begin: sampling sigma_n in the paper
		std::vector<int> sample_desired_hold_ids;
		if (initialstance_exists_in_tree)
		{
			//begin: sampling hand pos in the paper
			Vector3 rPoint = getRandomHandPose(m_future_reachable_hold_ids, m_reached_hold_ids, flag_greedy_search);
			//end: sampling hand pos in the paper

			float p_r = mSampler::getRandomBetween_01();
			if (p_r < 0.5f)
				sample_desired_hold_ids = sampleDesiredHoldsCenteringHand(rPoint, SimulationContext::BodyName::BodyRightArm);
			else
				sample_desired_hold_ids = sampleDesiredHoldsCenteringHand(rPoint, SimulationContext::BodyName::BodyLeftArm);
		}
		else
		{
			for (unsigned int i = 0; i < initial_stance.size(); i++)
			{
				sample_desired_hold_ids.push_back(initial_stance[i]);
			}
		}
		//end: sampling sigma_n in the paper

		printf("2");
		// ////////////////////////////////////////////////////////////////
		// because of distance metric of the current sampling (4 holds)  //
		// and choosing one node that the body's arms and legs have min  //
		// distance to the selected sample we need to use some heuristic //
		// for choosing a set of returned nodes                          //
		//////////////////////////////////////////////////////////////// //
		//std::vector<Vector3> sample_desired_hold_points;
		//for (unsigned int i = 0; i < sample_desired_hold_ids.size(); i++)
		//{
		//	sample_desired_hold_points.push_back(mClimberSampler.getHoldPos(sample_desired_hold_ids[i]));
		//}
		//bool flag_search_inside_nodes = mClimberSampler.earlyAcceptOfSample(sample_desired_hold_ids, !initialstance_exists_in_tree);
		//// use heuristic to find a list of nodes close to the desired sample
		//std::vector<int> list_close_nodes;
		//int min_index = getChosenNodesCloseToSample(flag_search_inside_nodes, sample_desired_hold_ids, sample_desired_hold_points, flag_greedy_search, list_close_nodes);
		//int chosen_index = 0;
		//if ((int)(list_close_nodes.size()) > min_index && min_index >= 0)
		//{
		//	chosen_index = list_close_nodes[min_index];
		//}
		//else
		//{
		//	mSampleStructure ret_sample;
		//	ret_sample.closest_node_index = chosen_index;
		//	for (unsigned int i = 0; i < sample_desired_hold_ids.size(); i++)
		//	{
		//		ret_sample.dPoint.push_back(mClimberSampler.getHoldPos(sample_desired_hold_ids[i]));
		//	}
		//	return ret_sample;
		//}
		//mSampleStructure nSample = mClimberSampler.getSampleToward(&mRRTNodes[chosen_index], sample_desired_hold_ids, !initialstance_exists_in_tree);
		//if (flag_greedy_search && nSample.isSet)
		//{
		//	int init_from = 0;
		//	if (initialstance_exists_in_tree)
		//	{
		//		init_from = index_node_initial_stance;
		//		if (init_from < 0)
		//			init_from = 0;
		//	}
		//	for (unsigned int i = init_from; i < mRRTNodes.size(); i++)
		//	{
		//		mNode node_i = mRRTNodes[i];
		//		// if we found a node representing a sample stance, we look for another node to go to sample stance if uniform growing is wanted
		//		bool flag_found_sample = mNode::isSetAEqualsSetB(node_i.cNodeInfo.hold_bodies_ids, nSample.desired_hold_ids);
		//		if (flag_found_sample)
		//		{
		//			return mClimberSampler.getSampleFrom(&mRRTNodes[i], nSample.desired_hold_ids, true); // just for showing and we are looking for another sample
		//		}
		//	}
		//}
		//nSample.to_hold_ids = sample_desired_hold_ids;
		//total_samples_num++;
		//return nSample;

		if (!flag_greedy_search)
		{
			printf("aa,");
			if (mTransitionFromTo.size() > 0)
			{
				float p_r = mSampler::getRandomBetween_01();
				if (p_r < 0.5) // rewire
				{
					Vector2 fTransition = mTransitionFromTo[0];
					mTransitionFromTo.erase(mTransitionFromTo.begin());

					mSampleStructure rSample = getSampleNodeToNode((int)fTransition[0], (int)fTransition[1]);
					return rSample;
				}
			}
			printf("ab,");
			float p_r = mSampler::getRandomBetween_01();
			if (p_r < 0.5f)
			{
				findBestPathToGoal(icurrent_case_study);
				int rIndex = path_nodes_indices[mSampler::getRandomIndex(path_nodes_indices.size())];
				if (rIndex > getInitFrom())
				{
					for (unsigned int i = 0; i < sample_desired_hold_ids.size(); i++)
					{
						sample_desired_hold_ids[i] = mRRTNodes[rIndex].cNodeInfo.hold_bodies_ids[i];
					}
				}
			}
			printf("ac,");
		}

		mSampleStructure nSample = getSampleNodeToHold(sample_desired_hold_ids, flag_greedy_search);
		nSample.to_hold_ids = sample_desired_hold_ids;
		printf("3");
		return nSample;

	}

	mSampleStructure getSampleNodeToHold(std::vector<int>& sample_desired_hold_ids, bool flag_greedy_search)
	{
		printf("\n ,sample: 1");
		std::vector<Vector3> sample_des_points;
		float mSizeDes = 0;
		Vector3 midDesPoint = mClimberSampler.getHoldStancePosFrom(sample_desired_hold_ids, sample_des_points, mSizeDes);

		printf(" 2");

		std::vector<float> value_nodes_stance;
		float sum_f = 0.0f;
		std::vector<int> list_nodes_n;
		std::vector<std::vector<int>> list_sample_n;
		for (unsigned int i = getInitFrom(); i < mRRTNodes.size(); i++)
		{
			mNode node_i = mRRTNodes[i];

			// if we found a node representing a sample stance, we look for another node to go to sample stance if uniform growing is wanted
			bool flag_found_sample = mNode::isSetAEqualsSetB(node_i.cNodeInfo.hold_bodies_ids, sample_desired_hold_ids);
			if (flag_found_sample)
			{
				if (flag_greedy_search)
				{
					list_nodes_n.clear();
					list_sample_n.clear();

					list_nodes_n.push_back(i);
					list_sample_n.push_back(sample_desired_hold_ids);

					return mClimberSampler.getSampleFrom(&mRRTNodes[i], sample_desired_hold_ids, true); // just for showing and we are looking for another sample
				}
				else
				{
					continue;
				}
			}

			std::vector<std::vector<int>> mListSamplei = mClimberSampler.getListOfSamples(&node_i, sample_desired_hold_ids, !initialstance_exists_in_tree);
			for (unsigned int n = 0; n < mListSamplei.size(); n++)
			{
				std::vector<int> sample_n = mListSamplei[n];

				/*if (node_i.isInTriedHoldSet(sample_n))
				{
					continue;
				}*/

				std::vector<Vector3> sample_n_points;
				float mSize = 0;
				Vector3 midPoint = mClimberSampler.getHoldStancePosFrom(sample_n, sample_n_points, mSize);

				float effect_mid_point = 0;
				if (mSize > 0)
				{
					effect_mid_point = 1;
				}

				float node_cost = getNodeCost(&node_i, mCaseStudy::movingLimbs); // it is calculated when a node is added
				float node_match_cost = sqrt(node_i.getSumDisEndPosTo(sample_desired_hold_ids, sample_des_points));
				float transition_cost = (sqrt(node_i.getSumDisEndPosTo(sample_n, sample_n_points)) // distance between node_i and stance n
					+ effect_mid_point * (node_i.cNodeInfo.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk).z() - midPoint.z()) // distance upward
					+ node_i.getCostDisconnectedArmsLegs() // matching cost
					+ node_i.getCostNumMovedLimbs(sample_n)); // cost for number of limbs moved
				float destination_cost = mClimberSampler.getSampleCost(sample_desired_hold_ids, sample_n);

				float f = 1 / (10.0f * node_cost + 1.0f * node_match_cost + 1.0f * transition_cost + destination_cost);

				createParticleList(10, sum_f, f, i, sample_n, value_nodes_stance, list_nodes_n, list_sample_n);
			}
		}

		printf(" 3");

		int min_index = chooseIndexFromParticles(value_nodes_stance, sum_f);

		printf(" 4");

		if (value_nodes_stance.size() > 0)
		{
			mNode cNode = mRRTNodes[list_nodes_n[min_index]];
			std::vector<int> cStance = list_sample_n[min_index];

			if (flag_greedy_search)
			{
				for (unsigned int i = getInitFrom(); i < mRRTNodes.size(); i++)
				{
					mNode node_i = mRRTNodes[i];

					// if we found a node representing a sample stance, we look for another node to go to sample stance if uniform growing is wanted
					bool flag_found_sample = mNode::isSetAEqualsSetB(node_i.cNodeInfo.hold_bodies_ids, cStance);
					if (flag_found_sample)
					{
						return mClimberSampler.getSampleFrom(&mRRTNodes[i], cStance, true); // just for showing and we are looking for another sample
					}
				}
			}

			mRRTNodes[list_nodes_n[min_index]].addTriedHoldSet(cStance);

			total_samples_num++;
			printf(" 5a");
			return mClimberSampler.getSampleFrom(&cNode, cStance, false);
		}

		printf(" 5b");
		return mSampleStructure();
	}

	mSampleStructure getSampleNodeToNode(int fNode, int tNode)
	{
		mSampleStructure sample_node_node = mClimberSampler.getSampleFrom(&mRRTNodes[fNode], &mRRTNodes[tNode]);
		sample_node_node.toNode = tNode;
		return sample_node_node;
	}

	//////////////////////////////////// utilities for RRT ////////////////////////////////////////////////

	int getBestNodeEnergy(std::vector<int>& tNodes, float& minCost)
	{
		int min_index = -1;
		for (unsigned int i = 0; i < tNodes.size(); i++)
		{
			float cCost = getNodeCost(&mRRTNodes[tNodes[i]], mCaseStudy::Energy);
			if (cCost < minCost)
			{
				minCost = cCost;
				min_index = i;
			}
		}
		return min_index;
	}

	void findBestPathToGoal(mCaseStudy icurrent_case_study)
	{
		Vector3 desiredPos = mContextRRT->getGoalPos();

		int min_index = -1;
		if (goal_nodes.size() == 0)
		{
			float minDis = FLT_MAX;
			for (unsigned int i = 0; i < mRRTNodes.size(); i++)
			{
				mNode nodei = mRRTNodes[i];

				//float cDis = getNodeCost(&nodei);
				float cDis = nodei.getSumDisEndPosTo(desiredPos);
				
				bool isHandConnectedToGoal = isNodeAttachedToGoal(&nodei);

				if (minDis > cDis)
				{
					minDis = cDis;
					min_index = i;
				}
			}

			cCostGoalMoveLimb = getNodeCost(&mRRTNodes[min_index], mCaseStudy::movingLimbs);
		}
		else
		{
			std::vector<int> _sameCostIndices;
			float minCost = FLT_MAX;
			float maxCost = -FLT_MAX;
			switch (icurrent_case_study)
			{
				case mCaseStudy::movingLimbs:
					for (unsigned int i = 0; i < goal_nodes.size(); i++)
					{
						float cCost = getNodeCost(&mRRTNodes[goal_nodes[i]], mCaseStudy::movingLimbs);
						if (cCost < minCost)
						{
							_sameCostIndices.clear();
							_sameCostIndices.push_back(i);
							minCost = cCost;
						}
						else if (cCost == minCost)
						{
							_sameCostIndices.push_back(i);
						}
					}

					cCostGoalMoveLimb = minCost;

					for (unsigned int i = 0; i < _sameCostIndices.size(); i++)
					{
						int index_in_goal_nodes = _sameCostIndices[i];
						float cCost = getNodeCost(&mRRTNodes[goal_nodes[index_in_goal_nodes]], mCaseStudy::Energy);
						if (cCost > maxCost)
						{
							maxCost = cCost;
							cGoalPathIndex = index_in_goal_nodes;
						}
					}

					cCostGoalControlCost = maxCost;

					break;
				case mCaseStudy::Energy:
					cGoalPathIndex = getBestNodeEnergy(goal_nodes, minCost);
					cCostGoalControlCost = minCost;
					cCostGoalMoveLimb = getNodeCost(&mRRTNodes[goal_nodes[cGoalPathIndex]], mCaseStudy::movingLimbs);
					break;
				default:
					break;
			}

			min_index = goal_nodes[cGoalPathIndex];
		}

		std::vector<int> cPath;
		mNode cNode = mRRTNodes[min_index];
		while (cNode.mFatherIndex != -1)
		{
			cPath.insert(cPath.begin(), cNode.nodeIndex);
			if (cNode.mFatherIndex != -1)
				cNode = mRRTNodes[cNode.mFatherIndex];
		}
		cPath.insert(cPath.begin(), cNode.nodeIndex);
		path_nodes_indices = cPath;

		return;
	}

	void playAnimation(mCaseStudy icurrent_case_study)
	{
		clock_t begin = clock();
		if (!mSample.restartSampleStartState)
			mSample.restartSampleStartState = true;

		if (!isPathFound)
		{
			findBestPathToGoal(icurrent_case_study);

			isPathFound = true;

			isAnimationFinished = false;
		}

		if (!isNodeShown)
		{
			if (lastPlayingNode < (int)path_nodes_indices.size() && path_nodes_indices[lastPlayingNode] < (int)mRRTNodes.size())
			{
				mController->startState = mRRTNodes[path_nodes_indices[lastPlayingNode]].cNodeInfo;
				isNodeShown = true;
				lastPlayingStateInNode = 0;
				lastPlayingNode++;
			}
			else
			{
				// restart showing animation
				lastPlayingNode = 0;
				cTimeElapsed = 0;
				isAnimationFinished = true;
			}
		}
		else
		{
			if (lastPlayingNode < (int)path_nodes_indices.size() && path_nodes_indices[lastPlayingNode] < (int)mRRTNodes.size())
			{
				if (lastPlayingStateInNode < (int)mRRTNodes[path_nodes_indices[lastPlayingNode]].statesFromFatherToThis.size())
				{
					mController->startState = mRRTNodes[path_nodes_indices[lastPlayingNode]].statesFromFatherToThis[lastPlayingStateInNode];
					lastPlayingStateInNode++;
				}
				else
				{
					isNodeShown = false;
				}
			}
			else
			{
				lastPlayingNode = 0;
				isNodeShown = false;
				cTimeElapsed = 0;
				isAnimationFinished = true;
			}
		}
//		Sleep(10); // 1 frame every 30 ms

		mController->takeStep();
		clock_t end = clock();
		cTimeElapsed += double(end - begin) / CLOCKS_PER_SEC;
		return;
	}

	void mAcceptOrRejectSample(mSampleStructure& iSample, targetMethod itargetMethd)
	{
		if (rejectCondition(iSample, itargetMethd))
		{
			iSample.isRejected = true;

			if (mController->startState.hold_bodies_ids[mController->startState.hold_bodies_ids.size() - 1] != -1 
				|| mController->startState.hold_bodies_ids[mController->startState.hold_bodies_ids.size() - 2] != -1)
			{
//				UpdateTree(mController->startState, iSample.closest_node_index, iSample.statesFromTo, iSample);
				UpdateTreeForGraph(mController->startState, iSample.closest_node_index, iSample.statesFromTo, iSample);
			}

			if (mNode::isSetAEqualsSetB(iSample.desired_hold_ids, mController->startState.hold_bodies_ids))
			{
				accepted_samples_num++;
			}

			itr_optimization_for_each_sample = 0;
			return;
		}
		if (acceptCondition(iSample))
		{
			iSample.isReached = true;
			accepted_samples_num++;
//			UpdateTree(mController->startState, iSample.closest_node_index, iSample.statesFromTo, iSample);
			UpdateTreeForGraph(mController->startState, iSample.closest_node_index, iSample.statesFromTo, iSample);
			itr_optimization_for_each_sample = 0;
			return;
		}
	}

	bool rejectCondition(mSampleStructure& iSample, targetMethod itargetMethd)
	{
		if (itargetMethd == targetMethod::Online)
		{
			if (iSample.numItrFixedCost > max_no_improvement_iterations)
				return true;
		}

		if (itr_optimization_for_each_sample > max_itr_optimization_for_each_sample)
			return true;

		if (iSample.isOdeConstraintsViolated)
			return true;

		if (iSample.initial_hold_ids[2] == -1 && iSample.initial_hold_ids[3] == -1)
			return false;

		if (mController->startState.hold_bodies_ids[2] == mController->startState.hold_bodies_ids[3] && mController->startState.hold_bodies_ids[2] == -1)
			return true;

		return false; // the sample is not rejected yet
	}

	bool acceptCondition(mSampleStructure& iSample)
	{
		for (unsigned int i = 0; i < iSample.desired_hold_ids.size(); i++)
		{
			if (mController->startState.hold_bodies_ids[i] != iSample.desired_hold_ids[i])
				return false; // not reached
		}

		Vector3 mid_point = mController->startState.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		float cDis = (mSample.destinationP[1] - mid_point).norm();
		if (cDis < 0.01f)
			return true; // is reached and accepted
		return false;
	}

	void m_Connect_Disconnect_ContactPoint(std::vector<int>& desired_holds_ids)
	{
		float min_reject_angle = (PI / 2) - (0.3f * PI);
		//float max_acceptable_angle = 1.3f * PI;

		for (unsigned int i = 0; i < desired_holds_ids.size(); i++)
		{
			if (desired_holds_ids[i] != -1)
			{
				Vector3 hold_pos_i = mClimberSampler.getHoldPos(desired_holds_ids[i]);
				Vector3 contact_pos_i = mController->startState.getEndPointPosBones(SimulationContext::ContactPoints::LeftLeg + i);

				float dis_i = (hold_pos_i - contact_pos_i).norm();

				if (dis_i < 0.25f * holdSize)
				{
					if (i <= 1) // left leg and right leg
					{
						Vector3 dir_contact_pos = -mController->startState.getBodyDirectionZ(SimulationContext::ContactPoints::LeftLeg + i);
						//mController->startState.bodyStates[SimulationContext::ContactPoints::LeftLeg + i].pos;
						float m_angle_btw = SimulationContext::getAbsAngleBtwVectors(-Vector3(0.0f, 0.0f, 1.0f), dir_contact_pos);

						if (m_angle_btw > min_reject_angle)
						{
							isReachedLegs[i] = true;
							if (i == 0 && mController->startState.hold_bodies_ids[1] > -1 && mController->startState.hold_bodies_ids[1] == desired_holds_ids[0])
							{
								continue;
							}
							if (i == 1 && mController->startState.hold_bodies_ids[0] > -1 && mController->startState.hold_bodies_ids[0] == desired_holds_ids[1])
							{
								continue;
							}
							mController->startState.hold_bodies_ids[i] = desired_holds_ids[i];
						}
						else
						{
							isReachedLegs[i] = false;
							mController->startState.hold_bodies_ids[i] = -1;
						}
					}
					else
					{
						mController->startState.hold_bodies_ids[i] = desired_holds_ids[i];
					}
				}
				else if (dis_i > 0.5f * holdSize)
				{
					if (i <= 1)
					{
						isReachedLegs[i] = false;
					}
					mController->startState.hold_bodies_ids[i] = -1;
				}
			}
			else
			{
				if (i <= 1)
				{
					isReachedLegs[i] = false;
				}
				mController->startState.hold_bodies_ids[i] = -1;
			}
		}
	}

	void mSteerFunc(bool isOnlineOptimization)
	{
		mController->loadPhysicsToMaster(itr_optimization_for_each_sample < max_waste_interations || !isOnlineOptimization);
		/*if (optimizerType==otCMAES && !useOfflinePlanning)
			Debug::throwError("CMAES cannot be used in online mode!");*/
		if (optimizerType==otCPBP)	
		{
			if (itr_optimization_for_each_sample==0 && useOfflinePlanning)
				mController->flc.reset();
			mController->optimize_the_cost(isOnlineOptimization, mSample.sourceP, mSample.destinationP, mSample.destinationA);
		}
		else 
			mController->optimize_the_cost_cmaes(itr_optimization_for_each_sample==0,mSample.sourceP, mSample.destinationP, mSample.destinationA);

		// check changing of the cost
		float costImprovement = std::max(0.0f,mSample.cOptimizationCost-mController->current_cost);
		rcPrintString("Traj. cost improvement: %f",costImprovement);
		if (costImprovement < noCostImprovementThreshold)
		{
			mSample.numItrFixedCost++;
		}
		else
		{
			mSample.numItrFixedCost = 0;
			
		}
		mSample.cOptimizationCost = mController->current_cost;

		//apply the best control to get the start state of next frame
		//advance_time is ture for the online optimization
		if (isOnlineOptimization) 
		{
			mStepOptimization(0, isOnlineOptimization);
		}
		return;
	}

	void mStepOptimization(int cTimeStep, bool debugPrint = false)
	{
		std::vector<BipedState> nStates;
		
		bool flagAddSimulation = mController->advance_simulation(cTimeStep, debugPrint,
							itr_optimization_for_each_sample >= max_waste_interations, nStates);

		for (unsigned int ns = 0; ns < nStates.size(); ns++)
		{
			BipedState nState = nStates[ns];
			mSample.statesFromTo.push_back(nState);
		}
		mSample.control_cost += mController->current_cost_control;

		if (!flagAddSimulation)
		{
			mSample.isOdeConstraintsViolated = true;
		}
	}

	void setArmsLegsHoldPoses()
	{
		mController->startState.hold_bodies_ids = mSample.initial_hold_ids;
		for (unsigned int i = 0; i < mController->startState.hold_bodies_ids.size(); i++)
		{
			if (mController->startState.hold_bodies_ids[i] != -1)
			{
				Vector3 hPos_i = mClimberSampler.getHoldPos(mController->startState.hold_bodies_ids[i]);
				Vector3 endPos_i = mController->startState.getEndPointPosBones(SimulationContext::BodyName::BodyLeftLeg + i);
				float cDis = (hPos_i - endPos_i).norm();
				if (cDis > 0.5f * holdSize)
				{
					mController->startState.hold_bodies_ids[i] = -1;
				}
			}
		}
		return;
	}

	void updateFutureReachedSet(BipedState& c, mSampleStructure& iSample)
	{
		/////////////// Begin Of: check which holds are reached by this body posture of this node
		for (unsigned int i = 0; i < m_future_reachable_hold_ids.size(); i++)
		{
			int k = m_future_reachable_hold_ids[i];
			bool flag_add = false;
			for (unsigned int  j = 0; j < mClimberSampler.indices_lower_than[k].size() && !flag_add; j++)
			{
				if (!mSampler::isInSetHoldIDs(mClimberSampler.indices_lower_than[k][j], m_reached_hold_ids))
					flag_add = true;
			}
			if (!flag_add)
			{
				m_future_reachable_hold_ids.erase(m_future_reachable_hold_ids.begin() + i);
				i--;
			}
		}

		bool flag_add_to_reached_holds = true;
		for (unsigned int i = 0; i < c.hold_bodies_ids.size() && flag_add_to_reached_holds; i++)
		{
			if (iSample.isSet)
			{
				if (c.hold_bodies_ids[i] != iSample.desired_hold_ids[i])
				{
					flag_add_to_reached_holds = false;
				}
			}
			if (c.hold_bodies_ids[i] == -1)
			{
				flag_add_to_reached_holds = false;
			}
		}

		if (c.hold_bodies_ids[0] == c.hold_bodies_ids[1] == c.hold_bodies_ids[2] == c.hold_bodies_ids[3])
		{
			flag_add_to_reached_holds = false;
		}

		Vector3 trunk_pos = c.getEndPointPosBones(SimulationContext::BodyName::BodyTrunk);
		for (unsigned int i = 0; i < c.hold_bodies_ids.size() && flag_add_to_reached_holds; i++)
		{
			if (c.hold_bodies_ids[i] != -1)
			{
				Vector3 hold_i = mClimberSampler.getHoldPos(c.hold_bodies_ids[i]);
				float m_angle = SimulationContext::getAngleBtwVectorsXZ(Vector3(1.0f,0.0f,0.0f), hold_i - trunk_pos);
				if (m_angle < 0 && m_angle > -PI)
				{
					bool flag_is_added = mSampler::addToSetHoldIDs(c.hold_bodies_ids[i], m_reached_hold_ids);
					if (flag_is_added)
					{
						// add the whole set of reachable holds
						int k = c.hold_bodies_ids[i];
						for (unsigned int  j = 0; j < mClimberSampler.indices_higher_than[k].size(); j++)
						{
							if (!mSampler::isInSetHoldIDs(mClimberSampler.indices_higher_than[k][j], m_reached_hold_ids))
								mSampler::addToSetHoldIDs(mClimberSampler.indices_higher_than[k][j], m_future_reachable_hold_ids);
						}
					}
					//
				}
			}
		}
		/////////////// End Of: check which holds are reached by this body posture of this node
		return;
	}

	////////////////////////////////// utilities for RRT nodes////////////////////////////////////////////

	std::vector<double> getNodeKeyFrom(BipedState& c)
	{
		std::vector<double> rKey;

		for (unsigned int i = 0; i < c.bodyStates.size(); i++)
		{
			BodyState body_i = c.bodyStates[i];
			rKey.push_back(body_i.getPos().x());
			rKey.push_back(body_i.getPos().y());
			rKey.push_back(body_i.getPos().z());
			//rKey.push_back(body_i.angle);

			//rKey.push_back(body_i.vel.x);
			//rKey.push_back(body_i.vel.y);
			//rKey.push_back(body_i.aVel);
		}

		return rKey;
	}

	int getNearestNode(BipedState& c)
	{
		int index_node = mKDTreeNodeIndices.nearest(getNodeKeyFrom(c));

		/*float minDis = FLT_MAX;
		int min_index = -1;
		for (unsigned int i = 0; i < mRRTNodes.size(); i++)
		{
			mNode node_i = mRRTNodes[i];
			float cDis = node_i.getDisFrom(c);
			if (cDis < minDis)
			{
				minDis = cDis;
				min_index = i;
			}
		}

		if (min_index != index_node)
		{
			int notifyme = 1;
		}*/

		return index_node;
	}

	float getNodeCost(mNode* iNode, mCaseStudy icurrent_case_study)
	{
		int cIndex = iNode->nodeIndex;
		float cCost = 0.0f;

		int mCounter = 0;

		while (cIndex >= 0)
		{
			switch (icurrent_case_study)
			{
			case mCaseStudy::movingLimbs:
				cCost += mRRTNodes[cIndex].cCost;
				break;
			case mCaseStudy::Energy:
				cCost += mRRTNodes[cIndex].control_cost;
				break;
			}

			cIndex = mRRTNodes[cIndex].mFatherIndex;

			mCounter++;
			if (mCounter > (int)mRRTNodes.size())
			{
				break;
			}
		}

		return cCost;
	}

	bool isBodyStateEqual(mNode* nearest_node, BipedState& c)
	{
		float pos_err = 0.01f;
		float angle_err = 0.01f;
		float ang_vel_err = 0.01f;
		float vel_err = 0.01f;
		for (unsigned int i = 0; i < c.bodyStates.size(); i++)
		{
			if ((nearest_node->cNodeInfo.bodyStates[i].getPos() - c.bodyStates[i].getPos()).norm() > pos_err)
			{
	//			printf("position: %f", (nearest_node->cNodeInfo.bodyStates[i].getPos() - c.bodyStates[i].getPos()).norm());
				return false;
			}
			if ((nearest_node->cNodeInfo.bodyStates[i].getAngle() - c.bodyStates[i].getAngle()).norm() > angle_err)
			{
	//			printf("angle: %f", (nearest_node->cNodeInfo.bodyStates[i].getAngle() - c.bodyStates[i].getAngle()).norm());
				return false;
			}
			if ((nearest_node->cNodeInfo.bodyStates[i].getAVel() - c.bodyStates[i].getAVel()).norm() > ang_vel_err)
			{
	//			printf("%f", (nearest_node->cNodeInfo.bodyStates[i].getAngle() - c.bodyStates[i].getAngle()).norm());
				return false;
			}
			if ((nearest_node->cNodeInfo.bodyStates[i].getVel() - c.bodyStates[i].getVel()).norm() > vel_err)
			{
		//		printf("%f", (nearest_node->cNodeInfo.bodyStates[i].getAngle() - c.bodyStates[i].getAngle()).norm());
				return false;
			}
		}
		return true;
	}

	void AddToTransitions(Vector2 nT)
	{
		bool flag_exists = false;
		for (unsigned int j = 0; j < mTransitionFromTo.size() && !flag_exists; j++)
		{
			Vector2 oT = mTransitionFromTo[j];
			if ((oT - nT).norm() < 0.1f)
			{
				flag_exists = true;
			}
		}
		if (!flag_exists)
		{
			mTransitionFromTo.push_back(nT);
			numPossRefine++;
		}
	}

	void rewire(mNode* node_nearest, int iFatherIndex, BipedState& c, std::vector<BipedState>& _fromFatherToThis)
	{
		node_nearest->cNodeInfo = c.getNewCopy(node_nearest->cNodeInfo.saving_slot_state, mContextRRT->getMasterContextID());
		int oFatherIndex = node_nearest->mFatherIndex;

		if (oFatherIndex != -1)
		{
			mSampler::removeFromSetHoldIDs(node_nearest->nodeIndex, mRRTNodes[oFatherIndex].mChildrenIndices);
		}

		node_nearest->mFatherIndex = iFatherIndex;
		node_nearest->cCost = node_nearest->getCostNumMovedLimbs(mRRTNodes[iFatherIndex].cNodeInfo.hold_bodies_ids);
		

		node_nearest->statesFromFatherToThis = _fromFatherToThis;
		if (iFatherIndex != -1)
		{
			mSampler::addToSetHoldIDs(node_nearest->nodeIndex, mRRTNodes[iFatherIndex].mChildrenIndices);
		}
				
		return;
	}

	void UpdateTreeForGraph(BipedState& c, int iFatherIndex, std::vector<BipedState>& _fromFatherToThis, mSampleStructure& iSample)
	{
		//bool flag_add_new_node = true;//false
		//if (mTriedPathIndicies.size() == 0)
		//{
		//	flag_add_new_node = true;
		//	
		//}
		//else
		//{
		//	int index_on_path = mTriedPathIndicies[mTriedPathIndicies.size() - 1];
		//	/*if (!isItReachedFromToOnPath[index_on_path - 1])
		//	{
		//		flag_add_new_node = true;
		//	}*/
		//	if (isItReachedFromToOnPath[index_on_path - 1])
		//	{
		//		flag_add_new_node = false;
		//	}
		//}
		//
		//if (!flag_add_new_node)
		//{
		//	return;
		//}
		
		mNode* closest_node = &mRRTNodes[getNearestNode(c)];
		if (isBodyStateEqual(closest_node, c)) //then if they are actually equal, rewire
		{
			iSample.toNode = closest_node->nodeIndex;
			return;
		}

		if (!initialstance_exists_in_tree)
		{
			if (mNode::isSetAEqualsSetB(initial_stance, c.hold_bodies_ids))
			{
				initialstance_exists_in_tree = true;
			}
		}

		int nNodeIndex = addNode(c, iFatherIndex, _fromFatherToThis, iSample);

		mNode nNode = mRRTNodes[nNodeIndex];
		
		if (initialstance_exists_in_tree && index_node_initial_stance < 0)
		{
			index_node_initial_stance = mRRTNodes.size() - 1;
		}

		// check goal is reached or not
		if (!isGoalReached)
		{
			isGoalReached = isNodeAttachedToGoal(&nNode);
		}
		return;
	}

	void UpdateTree(BipedState& c, int iFatherIndex, std::vector<BipedState>& _fromFatherToThis, mSampleStructure& iSample)
	{
		std::vector<int> nHoldIds = c.hold_bodies_ids;
		if (isReachedLegs[0])
		{
			nHoldIds[0] = iSample.desired_hold_ids[0];
		}
		if (isReachedLegs[1])
		{
			nHoldIds[1] = iSample.desired_hold_ids[1];
		}

		float nCost = 0;
		if (iFatherIndex != -1)
		{
			nCost = mNode::getDiffBtwSetASetB(mRRTNodes[iFatherIndex].cNodeInfo.hold_bodies_ids, c.hold_bodies_ids) + getNodeCost(&mRRTNodes[iFatherIndex], mCaseStudy::movingLimbs);
		}

		std::vector<int> nodes_with_same_stance;
		float minCost = FLT_MAX;
		int indexMinCost = -1;

		float minDist = FLT_MAX;
		int indexMinDist = -1;
		for (unsigned int i = getInitFrom(); i < mRRTNodes.size(); i++)
		{
			mNode node_i = mRRTNodes[i];
			if (node_i.isNodeEqualTo(nHoldIds))
			{
				float ci = getNodeCost(&mRRTNodes[i], mCaseStudy::movingLimbs);
				if (ci < minCost)
				{
					minCost = ci;
					indexMinCost = i;
				}

				float di = node_i.getDisFrom(c);
				if (di < minDist)
				{
					minDist = di;
					indexMinDist = i;
				}

				nodes_with_same_stance.push_back(i);
			}
		}

		bool flag_add_new_node = false;
		if (indexMinCost == -1 || indexMinDist == -1) // add if there is no node representing the stance
		{
			flag_add_new_node = true;
		}

		if (!flag_add_new_node && indexMinDist != -1) // add if in the same stance the dist is greater than threshold
		{
			if (minDist > 2 * boneLength)
			{
				flag_add_new_node = true;
			}
		}

		if (!flag_add_new_node && indexMinCost != -1) // add if whithin threshold we have a node with better cost
		{
			if (nCost < minCost)
			{
				flag_add_new_node = true;
			}
		}

		if (nCost < minCost && indexMinCost > -1)
		{
			numRefine++;
		}
		
		// do something about better cost
		int index_from_rewiring = mRRTNodes.size();
		if (nCost < minCost && indexMinCost > -1) // if there is possibility of equality and cost got better
		{
			mNode* closest_node = &mRRTNodes[indexMinCost];
			if (isBodyStateEqual(closest_node, c)) //then if they are actually equal, rewire
			{
				rewire(closest_node, iFatherIndex, c, _fromFatherToThis);
				nCost = closest_node->cCost;
				index_from_rewiring = closest_node->nodeIndex;

				if (iSample.toNode == -1)
				{
					numPossRefine++;
				}
				numRefine++;
				flag_add_new_node = false; // do not add if node got rewired
			}

			for (unsigned int i = 0; i < nodes_with_same_stance.size(); i++)
			{
				mNode ni = mRRTNodes[nodes_with_same_stance[i]];
				float ci = getNodeCost(&ni, mCaseStudy::movingLimbs);
				float cn = nCost + mNode::getDiffBtwSetASetB(c.hold_bodies_ids, ni.cNodeInfo.hold_bodies_ids);
				if (cn < ci)
				{
					AddToTransitions(Vector2(index_from_rewiring, ni.nodeIndex));
					for (unsigned int c = 0; c < ni.mChildrenIndices.size(); c++)
					{
						mNode node_c = mRRTNodes[ni.mChildrenIndices[c]];
						AddToTransitions(Vector2(index_from_rewiring, node_c.nodeIndex));
					}
				}
			}
		}

		if (!flag_add_new_node)
		{
			return;
		}

		updateFutureReachedSet(c, iSample);

		if (!initialstance_exists_in_tree)
		{
			if (mNode::isSetAEqualsSetB(initial_stance, c.hold_bodies_ids))
			{
				initialstance_exists_in_tree = true;
			}
		}

		int nNodeIndex = addNode(c, iFatherIndex, _fromFatherToThis, iSample);

		mNode nNode = mRRTNodes[nNodeIndex];
		
		if (initialstance_exists_in_tree && index_node_initial_stance < 0)
		{
			index_node_initial_stance = mRRTNodes.size() - 1;
		}

		// check goal is reached or not
		if (!isGoalReached)
		{
			isGoalReached = isNodeAttachedToGoal(&nNode);
		}
		return;
	}

	int addNode(BipedState& c, int iFatherIndex, std::vector<BipedState>& _fromFatherToThis, mSampleStructure& iSample)
	{
		mNode nNode = mNode(c.getNewCopy(mContextRRT->getNextFreeSavingSlot(), mContextRRT->getMasterContextID()), iFatherIndex, mRRTNodes.size(), _fromFatherToThis);

		nNode._goalNumNodeAddedFor = goal_nodes.size();
		nNode.control_cost = iSample.control_cost;

		if (iFatherIndex != -1)
		{
			nNode.cCost = nNode.getCostNumMovedLimbs(mRRTNodes[iFatherIndex].cNodeInfo.hold_bodies_ids);
		}

		nNode.poss_hold_ids = c.hold_bodies_ids;
		if (iSample.desired_hold_ids.size() > 0)
		{
			if (isReachedLegs[0])
			{
				nNode.poss_hold_ids[0] = iSample.desired_hold_ids[0];
			}
			if (isReachedLegs[1])
			{
				nNode.poss_hold_ids[1] = iSample.desired_hold_ids[1];
			}
		}

		mRRTNodes.push_back(nNode);

		if (iFatherIndex != -1)
		{
			mRRTNodes[iFatherIndex].mChildrenIndices.push_back(mRRTNodes.size() - 1);
		}

		mKDTreeNodeIndices.insert(getNodeKeyFrom(c), mRRTNodes.size() - 1);

		if (isNodeAttachedToGoal(&nNode))
		{
			int cNodeIndex = nNode.nodeIndex;
			while (cNodeIndex != -1)
			{
				mNode& cNode = mRRTNodes[cNodeIndex];
				if (mNode::isSetAEqualsSetB(cNode.cNodeInfo.hold_bodies_ids, initial_stance))
				{
					mSampler::addToSetHoldIDs(nNode.nodeIndex, goal_nodes);
					break;
				}
				cNodeIndex = cNode.mFatherIndex;
			}
		}

		iSample.toNode = nNode.nodeIndex;

		return nNode.nodeIndex;
	}

	bool isNodeAttachedToGoal(mNode* nodei)
	{
		if (nodei->cNodeInfo.hold_bodies_ids[3] != -1)
		{
			Vector3 rightHandPos = mClimberSampler.getHoldPos(nodei->cNodeInfo.hold_bodies_ids[3]);
			if ((mContextRRT->getGoalPos() - rightHandPos).norm() < 0.3f * holdSize)
			{
				return true;
			}
		}
		if (nodei->cNodeInfo.hold_bodies_ids[2] != -1)
		{
			Vector3 leftHandPos = mClimberSampler.getHoldPos(nodei->cNodeInfo.hold_bodies_ids[2]);
			if ((mContextRRT->getGoalPos() - leftHandPos).norm() < 0.3f * holdSize)
			{
				return true;
			}
		}
		return false;
	}

	/////////////////////////////////////////// common utilities //////////////////////////////////////////

	Vector3 uniformRandomPointBetween(Vector3 iMin, Vector3 iMax)
	{
		//Random between Min, Max
		float r1 = mSampler::getRandomBetween_01();
		float r2 = mSampler::getRandomBetween_01();
		float r3 = mSampler::getRandomBetween_01();
		Vector3 dis = iMax - iMin;

		return iMin + Vector3(dis.x() * r1, dis.y() * r2, dis.z() * r3); 
	}

}* myRRTPlanner;

bool advance_time;
bool play_animation;
bool FLAG_END_PROGRAM;

std::vector<float> timeOfFindingMotions;

double total_time_elasped = 0.0f;
mCaseStudy current_case_study = mCaseStudy::movingLimbs;
//bool _searchDoneForCaseStudy = false;

int _numShownTreeForGoal = 0;
double _counterShowing = 0.0f;

int cBodyNum;
int cAxisNum;
float dAngle;

mEnumTestCaseClimber TestID = mEnumTestCaseClimber::TestAngle; // 0, 1
bool revertToLastState;
bool holdBodyindexT;
int holdBodyindexI;
int currentIndexBody;

void forwardSimulation(float _time)
{
	if (!testClimber)
	{
		myRRTPlanner->mRunPathPlanner(
			useOfflinePlanning ? mRRT::targetMethod::Offiline : mRRT::targetMethod::Online, 
			advance_time, 
			play_animation, 
			(mCaseStudy)(max<int>((int)current_case_study, 0)), _time);
	}
	else
	{
		switch (TestID)
		{
		case TestAngle:
			if (revertToLastState)
			{
				revertToLastState = !revertToLastState;
				mCPBP->startState = mCPBP->resetState;
				mCPBP->loadPhysicsToMaster(true);
			}
			mContext->setMotorAngle(cBodyNum, cAxisNum, dAngle);
			rcPrintString("axis: %d, angle: %f \n",cAxisNum, dAngle);
			stepOde(timeStep,false);
			break;
		case TestCntroller:
			mTestClimber->runTest(advance_time, revertToLastState);
			if (holdBodyindexT)
			{
				if (holdBodyindexI < 0)
					holdBodyindexI = 0;
				if (holdBodyindexI > 3)
					holdBodyindexI = 3;
				currentIndexBody = holdBodyindexI;
				rcPrintString("body: %d\n", holdBodyindexI);
			}
			else
			{
				mTestClimber->desired_holds_ids[currentIndexBody] = holdBodyindexI;
				rcPrintString("hold: %d\n", holdBodyindexI);
			}
			break;
		default:
			break;
		}
		
	}
}


void EXPORT_API rcInit()
{
//	workerThreadManager=new WorkerThreadManager<int>(16);
//void main(int argc, char **argv)
//{
	char buff[100];
	sprintf_s(buff, "settings_climber.txt");
		
	FILE* rwFileStream;
	std::string mfilename = buff;
		
	fopen_s(&rwFileStream , mfilename.c_str(), "r");

	if (rwFileStream)
	{
		char buff2[100];
		fgets(buff2, 100, rwFileStream);
		sscanf_s(buff2, "%d,%d,%d", &nRows, &nColumns, &nTestNum);
		fclose(rwFileStream);
	}
	nRows = 4; nColumns = 1;
	srand(time(NULL));

	advance_time = true;
	FLAG_END_PROGRAM = false;

	revertToLastState = false;
	holdBodyindexT = true;
	holdBodyindexI = 0; // first by body

	mContext = new SimulationContext(testClimber, TestID, mDemoTestClimber::DemoRoute1, nRows, nColumns);
	
	BipedState startState;
	if (!testClimber)
	{
		startState.hold_bodies_ids.push_back(-1); // 0
		startState.hold_bodies_ids.push_back(-1); // 1
		startState.hold_bodies_ids.push_back(-1); // 2
		startState.hold_bodies_ids.push_back(-1); // 3
	}
	else
	{
		startState.hold_bodies_ids.push_back(-1);
		startState.hold_bodies_ids.push_back(-1);
		startState.hold_bodies_ids.push_back(-1);
		startState.hold_bodies_ids.push_back(-1);
	}
	startState.saving_slot_state = mContext->getMasterContextID();

	mCPBP = new robot_CPBP(mContext, startState);
	
	myRRTPlanner = new mRRT(mContext, mCPBP, nColumns);
	
	mTestClimber = new mTestControllerClass(mContext, mCPBP);

	if (testClimber)
	{
		switch (TestID)
		{
		case mEnumTestCaseClimber::TestAngle:
			cBodyNum = 0;
			cAxisNum = 0;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
//			dsMyToggleShadow();
			break;
		default:
			cBodyNum = 0;
			break;
		} 
	}
	else
	{
		cBodyNum = SimulationContext::BodyName::BodyRightArm;
	}

	play_animation = false;

//	runSimulation(argc, argv);

	dAllocateODEDataForThread(dAllocateMaskAll);

}

void EXPORT_API rcGetClientData(RenderClientData &data)
{
	data.physicsTimeStep = timeStep;
	data.defaultMouseControlsEnabled = true;
	data.maxAllowedTimeStep = timeStep;
}

void EXPORT_API rcUninit()
{
//	delete workerThreadManager;
	delete mContext;
	delete mCPBP;
	delete myRRTPlanner;
	delete mTestClimber;
}

void EXPORT_API rcUpdate()
{
	// simulation part
	rcPrintString("Total CPU time used: %f \n", total_time_elasped);
	startPerfCount();
//	_numShownTreeForGoal = (int)myRRTPlanner->goal_nodes.size();
//	if (_numShownTreeForGoal == (int)myRRTPlanner->goal_nodes.size())
//	{
	if (!pause)
	{
		forwardSimulation(total_time_elasped);
		// simulation part
		rcPrintString("Total paths found to goal: %d \n", (int)myRRTPlanner->goal_nodes.size());

		if (myRRTPlanner->goal_nodes.size() > timeOfFindingMotions.size())
			timeOfFindingMotions.push_back(total_time_elasped);
	}
	mContext->mDrawStuff(cBodyNum);
//		myRRTPlanner->drawTree();
//	}
	/*else
	{
		mContext->mDrawStuff(cBodyNum,false,false);
		myRRTPlanner->drawTree();

		if (_counterShowing > 10)
		{
			_counterShowing = 0;
			_numShownTreeForGoal = (int)myRRTPlanner->goal_nodes.size();
		}
	}*/
	static bool firstRun=true;
	if (firstRun)
	{
		rcSetLightPosition(lightXYZ[0],lightXYZ[1],lightXYZ[2]);
		rcSetViewPoint(xyz[0],xyz[1],xyz[2],lookAt[0],lookAt[1],lookAt[2]);
		firstRun=false;
	}

	if ((int)myRRTPlanner->goal_nodes.size() >= maxNumSimulatedPaths)
	{
		///////////////////////////////////////////////////////////////////////// second scenario dynamic graph vs static graph

		//char buff[100];
		//sprintf_s(buff, "TestFilesClimber\\FileLatticeHoldsCol%dRow%dT%d.txt", nColumns, nRows, nTestNum);
		//
		//FILE* rwFileStream;
		//std::string mfilename = buff;
		//
		//fopen_s(&rwFileStream , mfilename.c_str(), "w");

		//if (rwFileStream)
		//{
		//	for (unsigned int i = 0; i < timeOfFindingMotions.size(); i++)
		//	{
		//		if (i < timeOfFindingMotions.size() - 1)
		//			fprintf(rwFileStream, "%f,", timeOfFindingMotions[i]);
		//		else
		//			fprintf(rwFileStream, "%f", timeOfFindingMotions[i]);
		//	}
		//
		//	fclose(rwFileStream);
		//}

		//////////////////////////////////////////////////////////////// third scenario drawing tree

		//if (_numShownTreeForGoal == (int)myRRTPlanner->goal_nodes.size())
		//{
		//	if (flag_capture_video)
		//	{
		//		if (fileExists("out.mp4"))
		//			remove("out.mp4");
		//		system("screencaps2mp4.bat");
		//	}
		//	rcUninit();
		//	exit(0);
		//}

		///////////////////////////////////////////////////////////// first scenario studying energy vs movement

		if (!play_animation)
			play_animation = true;
		// simulation part
		rcPrintString("Current animation time:%f \n", myRRTPlanner->cTimeElapsed);
		switch (current_case_study)
		{
		case movingLimbs:
			rcPrintString("Path with min moving limbs among found paths: \n -Path %d with %d moving limbs, \n  and total control cost of %.*lf \n", 
				myRRTPlanner->cGoalPathIndex + 1, 
				(int)myRRTPlanner->cCostGoalMoveLimb, 
				2, 
				myRRTPlanner->cCostGoalControlCost);
			break;
		case Energy:
			rcPrintString("Path with min control cost among found paths: \n -Path %d with %d moving limbs, \n  and total control cost of %.*lf \n"
				, myRRTPlanner->cGoalPathIndex + 1, 
				(int)myRRTPlanner->cCostGoalMoveLimb,
				2, 
				myRRTPlanner->cCostGoalControlCost);
			break;
		default:
			break;
		}

		if (myRRTPlanner->isAnimationFinished)
		{
			if ((int)current_case_study < maxCaseStudySimulation)
				current_case_study = (mCaseStudy)(current_case_study + 1);
			
			myRRTPlanner->isPathFound = false;
			myRRTPlanner->isAnimationFinished = false;
		}

		if ((int)current_case_study >= maxCaseStudySimulation)
		{
			if (flag_capture_video)
			{
				if (fileExists("out.mp4"))
					remove("out.mp4");
				system("screencaps2mp4.bat");
			}
			rcUninit();
			exit(0);
		}
	}


	if (advance_time && !pause)
	{
		//if (_numShownTreeForGoal == (int)myRRTPlanner->goal_nodes.size())
	//	{
		total_time_elasped += getDurationMs()/1000.0f;
		//}
	//	else
	//	{
	//		_counterShowing++;
	//	}
		if (flag_capture_video)
			rcTakeScreenShot();
	}



}

void EXPORT_API rcOnKeyUp(int key)
{

}

void EXPORT_API rcOnKeyDown(int cmd)
{
	switch (cmd) 
	{
	case 'Q':	
		rcUninit();
		exit(0);
		break;
	case 'q':
		if (testClimber)
		{
			holdBodyindexI = -1;
		}
		else
		{
			myRRTPlanner->cGoalPathIndex--;
			myRRTPlanner->isPathFound = false;
			if (myRRTPlanner->cGoalPathIndex < 0)
			{
				myRRTPlanner->cGoalPathIndex = 0;
			}
		}
		break;
	case 'e':
		myRRTPlanner->cGoalPathIndex++;
		myRRTPlanner->isPathFound = false;
		if (myRRTPlanner->cGoalPathIndex >= (int)myRRTPlanner->goal_nodes.size())
		{
			myRRTPlanner->cGoalPathIndex = myRRTPlanner->goal_nodes.size() - 1;
		}
		break;
	//case 'm':
	//	dsMyToggleMouse();
	//	break;
	case ' ':
		play_animation = !play_animation;
		break;
	case 'o':
		advance_time = !advance_time;
		break;
	case 'p':
		pause = !pause;
		break;
	case 'r':
		revertToLastState = !revertToLastState;
		break;
	case 't':
		holdBodyindexT = !holdBodyindexT;
		if (holdBodyindexT)
		{
			holdBodyindexI = currentIndexBody;
		}
		else
		{
			holdBodyindexI = mTestClimber->desired_holds_ids[currentIndexBody];
		}
		break;
	//case 'q':
	//	FLAG_END_PROGRAM = true;
	//	break;
	case 'z':
		if (testClimber)
		{
			cBodyNum += 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 'x':
		if (testClimber)
		{
			cBodyNum -= 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 'a':
		if (testClimber) 
		{
			cAxisNum += 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 's':
		if (testClimber)
		{
			cAxisNum -= 1;
			dAngle = mContext->getDesMotorAngle(cBodyNum, cAxisNum);
		}
		break;
	case 'c':
		if (testClimber)
		{
			dAngle += 0.1f;
		}
		break;
	case 'v':
		if (testClimber)
		{
			dAngle -= 0.1f;
		}
		break;
	default:
		if (testClimber)
		{
			if (cmd >= 48 && cmd <= 57)
			{
				holdBodyindexI = cmd - 48;
			}
		}
		break;
	}
	
}

void EXPORT_API rcOnMouse(float rayStartX, float rayStartY, float rayStartZ, float rayDirX, float rayDirY, float rayDirZ, int button, int x, int y)
{
	return;
}

//void runSimulation(int argc, char **argv)
//{
//	// setup pointers to drawstuff callback functions
//	dsFunctions fn;
//	fn.version = DS_VERSION;
//	fn.start = &start;
//	fn.step = &simLoop;
//	fn.command = &command;
//	fn.stop = 0;
//	fn.path_to_textures = DRAWSTUFF_TEXTURE_PATH;
//	dsSimulationLoop (argc,argv,1280,720, &fn);
//}

