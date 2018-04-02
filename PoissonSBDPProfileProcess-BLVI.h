
/********************


**********************/

#ifndef POISSONSBDPPROFILEVI_H
#define POISSONSBDPPROFILEVI_H

#include "PoissonDPProfileProcessVI.h"
#include "SBDPProfileProcessVI.h"
#include "Random.h"

// superclass for Poisson (F81) implementations
class PoissonSBDPProfileProcessVI: public virtual PoissonDPProfileProcessVI, public virtual SBDPProfileProcessVI	{

	public:

	PoissonSBDPProfileProcessVI() : InitIncremental(0) {}
	virtual ~PoissonSBDPProfileProcessVI() {}
        double maxPCAT;
        double GetmaxPCAT() {return maxPCAT;}
	virtual double Move(double tuning = 1, int n = 1, int nrep = 1)	{

		// totchrono.Start();
		for (int rep=0; rep<nrep; rep++)	{

			// incchrono.Start();
			GlobalUpdateParameters();
			GlobalUpdateSiteProfileSuffStat();
			UpdateModeProfileSuffStat();
			if ((!rep) && InitIncremental)	{
				cerr << "init incremental\n";
				InitIncremental--;
				IncrementalSampleAlloc();
				UpdateModeProfileSuffStat();
			}
			GlobalMixMove(5,1,0.001);
			// MoveOccupiedCompAlloc(5);
			// MoveAdjacentCompAlloc(5);
			// incchrono.Stop();
			GlobalUpdateParameters();
			GlobalUpdateSiteProfileSuffStat();
			MoveHyper(tuning,10);
		}
		// totchrono.Stop();
		return 1;
	}

	virtual double LogProxy(int site, int cat)	{
		return DiffLogSampling(cat,site);
	}

	protected:

	virtual void Create(int innsite, int indim)	{
		PoissonDPProfileProcessVI::Create(innsite,indim);
		SBDPProfileProcessVI::Create(innsite,indim);
                Gpcat = new double[GetNmodeMax()];
                PCAT = new double[GetNmodeMax()];
                for (int mode = 0; mode < GetNmodeMax(); mode++) {
                     Gpcat[mode] = 0;
                     // PCAT[mode] = rnd::GetRandom().Uniform();
                     PCAT[mode] = 0.001;
                } 
	}

	virtual void Delete()	{
		SBDPProfileProcessVI::Delete();
		PoissonDPProfileProcessVI::Delete();
	}

	double GlobalMixMove(int MCsamples, int nallocrep, double epsilon);
	void SlaveMixMove();

	double IncrementalDPMove(int nrep, double c)	{
		cerr << "error : in poisson sbdp incremental\n";
		exit(1);
		return 1;
	}
	double IncrementalDPMove(int nrep)	{
		cerr << "error : in poisson sbdp incremental\n";
		exit(1);
		return 1;
	}

	virtual void SwapComponents(int cat1, int cat2)	{
		SBDPProfileProcessVI::SwapComponents(cat1,cat2);
	}

	// virtual void ToStream(ostream& os);
	virtual void FromStream(istream& is)	{
		PoissonDPProfileProcessVI::FromStream(is);
		ResampleWeights();
	}

	int InitIncremental;
        // Monte Carlo for variational parameter 
        double* MCweight;
        double* MCdpcat;
        double* MCLogSamplingArray;
        double* MCLogQpcat;
        double* MCLogpcat;
        /*double MCdpcat;
        double MCLogSamplingArray;
        double MCLogQpcat;
        double MCLogpcat;*/

        double* MCfpcat;
        double* MChpcat;
        double* meanfpcat;
        double* meanhpcat;
        double* varpcat;
        double* covpcat;
        double* a_pcat;
        double* grad_pcat;
        double* Gpcat;
        double* PCAT;
        
        

};

#endif

