#pragma once
#include "MathUtils.h"
/*
 * See:
 * http://www.geometrictools.com/Documentation/KBSplines.pdf
 * http://www.gamedev.net/topic/500802-tcb-spline-interpolation/
 * */

namespace AaltoGames
{

class RecursiveTCBSpline
{
protected:
	float current, dCurrent;
    float c;   //kept at 0, because we need to have tangent continuity (incoming = outgoing) for the recursion to work as implemented
public:
    float savedCurrent,savedDCurrent;
    //Default tcb values 0 => catmull-rom spline
    float t;
    float b;
    float linearMix;
    RecursiveTCBSpline()
	{
        current=0;
        dCurrent=0;
        t=0;
        c=0;
        b=0;
        linearMix=0;
	}
    void setValueAndTangent(float current, float dCurrent)
    {
        this->current=current;
        this->dCurrent=dCurrent;
    }
    void setValue(float current)
    {
        this->current = current;
    }
    float getValue()
	{
        return current;
	}
    //incoming tangent at p1, based on three points p0,p1,p2 and their times t0,t1,t2
    float TCBIncomingTangent(float p0, float t0, float p1, float t1, float p2, float t2)
    {
     	return 0.5f*(1-t)*(1+c)*(1-b)*(p2-p1)/(t2-t1) + 0.5f*(1-t)*(1-c)*(1+b)*(p1-p0)/(t1-t0);
    }

    float TCBOutgoingTangent(float p0, float t0, float p1, float t1, float p2, float t2)
    {
     	return 0.5f*(1-t)*(1-c)*(1-b)*(p2-p1)/(t2-t1) + 0.5f*(1-t)*(1+c)*(1+b)*(p1-p0)/(t1-t0);
    }

    // p0 y coordinate of the current point
    // p1 y coordinate of the next point
    // s interpolation value, (t-t0)/timeDelta
    // to0 tangent out of (p0)
    // ti1 tangent in of (p1)
    float InterpolateHermitian(float p0, float p1, float s, float to0, float ti1, float timeDelta)
    {
        float s2=s*s;
        float s3=s2*s;
        //the tcb formula using a Hermite interpolation basis from Eberly
        return (2*s3-3*s2+1)*p0 + (-2*s3+3*s2)*p1 + (s3-2*s2+s)*timeDelta*to0 + (s3-s2)*timeDelta*ti1;
    }

    float InterpolateHermitianTangent(float p0, float p1, float s, float to0, float ti1, float timeDelta)
    {
        float s2=s*s;
        return (6 * s2 - 6 * s) * p0 + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 4 * s + 1) * timeDelta * to0 + (3 * s2 - 2 * s) * timeDelta * ti1;
    }

    //Step the curve forward using the internally stored current value-tangent pair current, dCurrent, and the next two points p1,p2 at times t1,t2. 
    //Note: the times are absolute times, e.g., t1=0.1f, t2=2.3f, not the relative distances of the control points in time.
    //
    void step(float timeStep, float p1, float t1, float p2, float t2)
    {
       /* if (t1==0)
            Debug::throwError("RecursiveTCBSpline::step() called with zero t1");
        if (t2==t1)
            Debug::throwError("RecursiveTCBSpline::step() called t1==t2");*/

        float newLinearVal = current + (p1 - current) * timeStep / t1;
        float newLinearTangent = (p1 - current) / t1;
        if (linearMix >= 1)
        {
            current = newLinearVal;
            dCurrent = newLinearTangent;
        }
        else
        {
            float p1IncomingTangent = TCBIncomingTangent(current, 0, p1, t1, p2, t2);
            float newTCBVal = InterpolateHermitian(current, p1, timeStep / t1, dCurrent, p1IncomingTangent, t1);
            float newTCBTangent = InterpolateHermitianTangent(current, p1, timeStep / t1, dCurrent, p1IncomingTangent, t1) / t1;
            current = linearMix * newLinearVal + (1.0f - linearMix) * newTCBVal;
            dCurrent = linearMix * newLinearTangent + (1.0f - linearMix) * newTCBTangent;
        }
    }
    void save()
    {
        savedCurrent = current;
        savedDCurrent = dCurrent;
    }
    void restore()
    {
        current = savedCurrent;
        dCurrent = savedDCurrent;
    }
    void copyStateFrom(RecursiveTCBSpline src)
    {
        current = src.current;
        dCurrent = src.dCurrent;
    }
};

} //AaltoGames