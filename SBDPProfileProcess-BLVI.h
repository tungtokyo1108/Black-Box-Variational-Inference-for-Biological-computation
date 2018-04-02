
/********************

**********************/


#ifndef SBDPPROFILEVI_H
#define SBDPPROFILEVI_H

#include <cmath>
#include "DPProfileProcessVI.h"


const int refnmodemaxVI = 5000;

// general superclass for all finite process mixtures on site-specific profiles
class SBDPProfileProcessVI: public virtual DPProfileProcessVI	{

	using MixtureProfileProcess::LogStatPrior;

	public:

	SBDPProfileProcessVI() : DPProfileProcessVI(), nmodemax(refnmodemaxVI), V(0), maxweighterror(0) {}
	virtual ~SBDPProfileProcessVI(){}

	protected:

	virtual void DrawProfileFromPrior();

	double GetMaxWeightError() {return maxweighterror;}
	void ResetMaxWeightError() {maxweighterror = 0;}

	virtual void Create(int innsite, int indim);
	virtual void Delete();

	virtual int GetNmodeMax() {return GetNsite() > nmodemax ? nmodemax : GetNsite();}
	virtual void SetNmodeMax(int n) {nmodemax = n;}

	virtual double IncrementalDPMove(int nrep, double epsilon) = 0;

	double IncrementalDPMove(int nrep)	{
		cerr << "inc move deactivated\n";
		exit(1);
	}

	double GetWeightEnt()	{
		double tot = 0;
		for (int k=0; k<GetNcomponent(); k++)	{
			if (weight[k] > 1e-8)	{
				tot -= weight[k] * log(weight[k]);
			}
		}
		return tot;
	}

	int GetNDisplayedComponent()	{
		return GetNOccupiedComponent();
	}

	int GetLastOccupiedComponent()	{
		int kmax = 0;
		for (int i=0; i<GetNsite(); i++)	{
			if (kmax < alloc[i])	{
				kmax = alloc[i];
			}
		}
		return kmax;
	}

	int GetNCutoff(double cutoff)	{
		int n = (int) (GetNOccupiedComponent() * (1 - cutoff));
		int k = GetLastOccupiedComponent();
		int tot = occupancy[k];
		while (k && (tot < n))	{
			k--;
			if (k)	{
				tot += occupancy[k];
			}
		}
		return k;
	}
		
	virtual void SwapComponents(int cat1, int cat2);

	// double LogAllocPrior();
	double LogIntegratedAllocProb();

	// void ShedTail();

	// redefined
	void SampleAlloc();

	void IncrementalSampleAlloc();

	void SampleWeights();
	void ResampleWeights();
        

	// void ResampleLastWeight();
	double MoveOccupiedCompAlloc(int nrep = 1);
	double MoveAdjacentCompAlloc(int nrep = 1);

	double LogStatPrior();

	/*
	double totweight;
	double cumulProduct;
	*/
       
	int nmodemax;
        double kappa_alpha;
        double Pkappa_alpha;
        double Pkappa;
	double* V;
	double* weight;
        double* V_P;
        double* weight_P;
        
        // Variational method
        void MCQVAlpha(int MCsamples);
        void MCQVBeta(int MCsamples);
        void GradVAlpha(int MCsamples);
        void GradVBeta(int MCsamples);

        double GetMCfVAlpha(int MCsamples) {return MCfVAlpha[MCsamples];}
        double GetMChVAlpha(int MCsamples) {return MChVAlpha[MCsamples];}
        double GetMCfVBeta(int MCsamples)  {return MCfVBeta[MCsamples];}
        double GetMChVBeta(int MCsamples)  {return MChVBeta[MCsamples];}
        double Getgrad_VAlpha()  {return grad_VAlpha;}
        double Getgrad_VBeta()  {return grad_VBeta;}
        double MoveVAlpha(int MCsamples);
        double MoveVBeta(int MCsamples);
        double MoveKappa(int MCsamples);

        // Monte Carlo parameter 
        double** MCV;
        double* MCDQVAlpha;
        double* MCLogIntegratedAllocProb;
        double* MCLogQV;
        double* MCLogV;
        double* MCfVAlpha;
        double* MChVAlpha;
        double meanfVAlpha;
        double meanhVAlpha;
        double varVAlpha;
        double covVAlpha;
        double a_VAlpha;
        
        double* MCDQVBeta;
        double* MCfVBeta;
        double* MChVBeta;
        double meanfVBeta;
        double meanhVBeta;
        double varVBeta;
        double covVBeta;
        double a_VBeta;

	double maxweighterror;
        double grad_VAlpha;
        double grad_VBeta;
        double GVAlpha;
        double GVBeta;
};

#endif

