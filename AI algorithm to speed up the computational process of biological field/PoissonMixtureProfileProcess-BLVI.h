
/********************


**********************/

#ifndef POISSONMIXTUREPROFILEVI_H
#define POISSONMIXTUREPROFILEVI_H

#include "PoissonProfileProcess.h"
#include "MixtureProfileProcess.h"

// superclass for Poisson (F81) implementations
class PoissonMixtureProfileProcessVI: public virtual PoissonProfileProcess, public virtual MixtureProfileProcess	{

	public:

	PoissonMixtureProfileProcessVI() : profilesuffstatcount(0) {}
	virtual ~PoissonMixtureProfileProcessVI() {}

	protected:

	virtual void Create(int innsite, int indim);
	virtual void Delete();

	virtual void CreateComponent(int k) {
		occupancy[k] = 0;
		/*
		int* catnsub = profilesuffstatcount[k];
		for (int i=0; i<GetDim(); i++)	{
			catnsub[i] = 0;
		}
		*/
		SampleStat(k);
	}
	virtual void DeleteComponent(int k) {
	}
	virtual void UpdateComponent(int k) {}

	// posterior
	// collects sufficient statistics across sites, pools them componentwise
	void UpdateModeProfileSuffStat();

	// virtual void CreateComponent(int k)	{}

	// suffstat lnL of all sites allocated to component cat
	double ProfileSuffStatLogProb(int cat);

	// difference between:
	// suffstat lnL of all sites allocated to component cat when site <site> is among them, and
	// suffstat lnL of all sites allocated to component cat when site <site> is not among them
	double DiffLogSampling(int cat, int site);
	virtual double LogStatProb(int site, int cat);
	double MoveDirWeights(int MCsamples);

	double MoveProfile();
	double MoveProfile(int cat);

	void SwapComponents(int cat1, int cat2);
	void AddSite(int site, int cat);
	void RemoveSite(int site, int cat);

	double GetNormRate(int k)	{

		double tot = 0;
		for (int i=0; i<GetDim(); i++)	{
			for (int j=i+1; j<GetDim(); j++)	{
				tot += profile[k][i] * profile[k][j];
			}
		}
		return 2*tot;
	}

	virtual double GetNormalizationFactor()	{
		UpdateOccupancyNumbers();
		double norm = 0;
		int tot = 0;
		for (int k=0; k<GetNcomponent(); k++)	{
			if (occupancy[k])	{
				double tmp = GetNormRate(k);
				norm += (occupancy[k] + 1) * tmp;
				tot += occupancy[k] + 1;
			}
		}
		/*
		if (tot != GetNsite() + GetNcomponent())	{
			cerr << "error in norm factor\n";
			cerr << tot << '\t' << GetNsite() << '\t' << GetNcomponent() << '\n';
			exit(1);
		}
		*/
		norm /= tot;
		return norm;
	}
       
        // Variational method
        void MCQdirweight(int MCsamples);
        void Graddirweight(int MCsamples);
        double MoveDirWeight(int MCsamples);
        
        double GetMCfdirweight(int MCsamples)  {return MCfdirweight[MCsamples];}
        double GetMChdirweight(int MCsamples)  {return MChdirweight[MCsamples];}
        double Getgrad_dirweight()  {return grad_dirweight;}
        
        // Monte Carlo parameter
        double* MCDQdirweight;
        double* MCLogStatProb;
        double* MCLogQdirweight;
        double* MCLogdirweight;
        double* MCfdirweight;
        double* MChdirweight;
        double*** MCProfile;
        double* Pdirweight;
        
        double meanfdirweight;
        double meanhdirweight;
        double vardirweight;
        double covdirweight;
        double a_dirweight;
        double grad_dirweight;
        double* Gdirweight;
        
	// private:
	int** profilesuffstatcount;
};

#endif

