
/********************

**********************/


#ifndef GAMRATEVI_H
#define GAMRATEVI_H

#include "RateProcess.h"
#include <gsl/gsl_sf_psi.h>

class GammaRateProcessVI : public virtual RateProcess {

	public:

	GammaRateProcessVI() : Ncat(0), rate(0) {}
	virtual ~GammaRateProcessVI() {}

	double GetAlpha() {return alpha;}

	int GetNrate(int site)	{
		if (SumOverRateAllocations())	{
			return Ncat;
		}
		return 1;
	}

	int GetNcat() {return Ncat;}

	double GetRate(int site, int cat = 0)	{
		// cat should be == 0
		if (SumOverRateAllocations())	{
			return rate[cat];
		}
		return rate[alloc[site]];
	}

	double GetRateWeight(int site, int cat)	{
		if (SumOverRateAllocations())	{
			return 1.0/Ncat;
		}
		return 1.0;
	}

	void ActivateSumOverRateAllocations() {
		sumflag = true;
	}

	void InactivateSumOverRateAllocations(int* ratealloc) {
		for (int i=0; i<GetNsite(); i++)	{
			alloc[i] = ratealloc[i];
		}
		sumflag = false;
	}

	double GetPriorMeanRate()	{
		double total = 0;
		for (int k=0; k<GetNcat(); k++)	{
			total += rate[k];
		}
		return total / GetNcat();
	}

        void SetAlpha(double inalpha)  {
             alpha = inalpha;
	     UpdateDiscreteCategories();
        }        

	void ToStream(ostream& os);
	void FromStream(istream& is);

	// protected:

	void Create(int innsite, int inncat);
	void Delete();

	void SampleRate();


	void GlobalUpdateRateSuffStat();
	void SlaveUpdateRateSuffStat();
	void UpdateRateSuffStat();
     
	double UpdateDiscreteCategories();
	
	int Ncat;
	double* rate;
	double Palpha;
        double alpha;
	int* alloc;
	int* ratesuffstatcount;
	double* ratesuffstatbeta;
        double GetRateSample(int GetNcat()) {return rate[GetNcat()];}
        double LogRatePrior();
        // Variational method
        void MCQRateAlpha(int MCsamples);
        void SlaveMCQRateAlpha();
        void GlobalMCQRateAlpha(int MCsamples);
        void GradQRateAlpha(int MCsamples);
        double GetMCfRateAlpha(int MCsamples) {return MCfRateAlpha[MCsamples];}
        double GetMChRateAlpha(int MCsamples) {return MChRateAlpha[MCsamples];}
        double Getgrad_RateAlpha()  {return grad_RateAlpha;}

        double MoveRateAlpha(int MCsamples);
        double Move(int MCsamples);

        void UpdateRateProcess();

        // Monte Carlo parameter
        double* MCDQRateAlpha;
        double* MCLogRateSuffStatProb;
        double* MCLogQRate;
        double* MCLogRate;
        double* MCfRateAlpha;
        double* MChRateAlpha;
        double meanfRateAlpha;
        double meanhRateAlpha;
        double varRateAlpha;
        double covRateAlpha;
        double a_RateAlpha;
        double grad_RateAlpha;
        double GRateAlpha; 


};

#endif

