
/********************


**********************/


#ifndef GAMMABRANCHVI_H
#define GAMMABRANCHVI_H

#include "TaxonSet.h"
#include "BranchProcess.h"
#include "RateProcess.h"
#include <gsl/gsl_sf_psi.h>
#include "Random.h"

class GammaBranchProcessVI : public virtual BranchProcess	{

	public:

	GammaBranchProcessVI() : betaprior(0) {}
	virtual ~GammaBranchProcessVI() {}
        
	double LogBranchLengthPrior(const Branch* branch);
        double LogPBranchLengthPrior(const Branch* branch);
        double DQBranchAlpha(const Branch* branch);
        double DQBranchBeta(const Branch* branch);        

	double GetBranchAlpha() {return branchalpha;}
	double GetBranchBeta() {return branchbeta;}
        double GetPBranchAlpha() {return Pbranchalpha;}
        double GetPBranchBeta() {return Pbranchbeta;}
        double Getgrad_branchalpha() {return grad_branchalpha;}
        double Getgrad_branchbeta() {return grad_branchbeta;}
        double GetMCfbranchalpha(int MCsamples) {return MCfbranchalpha[MCsamples];}
        double GetMChbranchalpha(int MCsamples) {return MChbranchalpha[MCsamples];} 
        double GetMCfbranchbeta(int MCsamples)  {return MCfbranchbeta[MCsamples];}
        double GetMChbranchbeta(int MCsamples)  {return MChbranchbeta[MCsamples];}      
        double Movebranchalpha(int MCsamples); 
        double Movebranchbeta(int MCsamples);
        double MoveLength();
        double Move(int MCsamples);

	void SampleLength();
	void SampleLength(const Branch* branch);
        void PSampleLength();
        void PSampleLength(const Branch* branch);
        
        /*void SlaveMCQbranchalpha();
        void GlobalMCQbranchalpha(int MCsamples);*/
        void MCQbranchalpha(int MCsamples);

        /*void SlaveMCQbranchbeta();
        void GlobalMCQbranchbeta(int MCsamples);*/
        void MCQbranchbeta(int MCsamples);

        void GradQbranchalpha(int MCsamples);
        void GradQbranchbeta(int Mcsamples);        
	void ToStreamWithLengths(ostream& os, const Link* from);

	void ToStream(ostream& os);
	void FromStream(istream& is);

	protected:

	virtual void Create(Tree* intree, double inalpha = 1, double inbeta = 10, double Pinalpha = 1, double Pinbeta = 10)	{
		BranchProcess::Create(intree);
		branchalpha = inalpha;
		branchbeta = inbeta;
                Pbranchalpha = Pinalpha;
                Pbranchbeta = Pinbeta;
                Gbranchalpha = rnd::GetRandom().Uniform(); 
                Gbranchbeta = rnd::GetRandom().Uniform(); 
		// RecursiveSampleLength(GetRoot());
	}

	virtual void Delete() {}
        
	double branchalpha;
	double branchbeta;
        double Pbranchalpha;
        double Pbranchbeta;
        

        // Monte Carlo simulation 
        
        double* MCblarray;
        double* MCDQbranchalpha;
        double* MCDQbranchbeta;
        double* MCLogLikelihoodBranch;
        double* MCLogQBranchLengthPrior;
        double* MCLogBranchLengthPrior;
        double* MCfbranchalpha;
        double* MChbranchalpha;
        double* MCfbranchbeta;
        double* MChbranchbeta;
        double meanfbranchalpha;
        double meanhbranchalpha;
        double varbranchalpha;
        double covbranchalpha;
        double a_branchalpha;
        double meanfbranchbeta;
        double meanhbranchbeta;
        double varbranchbeta;
        double covbranchbeta;
        double a_branchbeta;
   
        double grad_branchalpha;
        double grad_branchbeta;
        double Gbranchalpha;
        double Gbranchbeta;

	int betaprior;
        double dqbranchalpha;
        
};

#endif

