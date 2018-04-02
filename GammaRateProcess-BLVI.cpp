
/********************


**********************/


#include "GammaRateProcessVI.h"
#include "Random.h"
#include "IncompleteGamma.h"

#include <cassert>
#include "Parallel.h"
#include <string.h>

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
//	* GammaRateProcess
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------


void GammaRateProcessVI::Create(int innsite, int inncat)	{
	if (! rate)	{
		RateProcess::Create(innsite);
                Ncat = 4;
                alpha = 1;
                GRateAlpha = rnd::GetRandom().Uniform();
                Palpha = 1;
                rate = new double[GetNcat()];
		alloc = new int[GetNsite()];
		ratesuffstatcount = new int[GetNcat()];
		ratesuffstatbeta = new double[GetNcat()];
		// SampleRate();
	}
}


void GammaRateProcessVI::Delete() 	{
	delete[] rate;
	delete[] alloc;
	delete[] ratesuffstatcount;
	delete[] ratesuffstatbeta;
	rate = 0;
	alloc = 0;
	ratesuffstatcount = 0;
	ratesuffstatbeta = 0;
        RateProcess::Delete();
}

void GammaRateProcessVI::ToStream(ostream& os)	{
	os << alpha << '\n';
        
}

void GammaRateProcessVI::FromStream(istream& is)	{
        is >> alpha;
	SetAlpha(alpha);
}


// ------------------------------ Computate the variational distribution of GammaRateProcess -------------------------------------------------------------------

void GammaRateProcessVI::SampleRate()  {
     // alpha = rnd::GetRandom().sExpo();
     alpha = 1;
     UpdateDiscreteCategories(); 
}

double GammaRateProcessVI::UpdateDiscreteCategories()  {

        double* x = new double[GetNcat()];
	double* y = new double[GetNcat()];
	double lg = rnd::GetRandom().logGamma(alpha+1.0);
	for (int i=0; i<GetNcat(); i++)	{
		x[i] = PointGamma((i+1.0)/GetNcat(),alpha,alpha);
	}
	for (int i=0; i<GetNcat()-1; i++)	{
		y[i] = IncompleteGamma(alpha*x[i],alpha+1,lg);
	}
	y[GetNcat()-1] = 1.0;
	rate[0] = GetNcat() * y[0];
	for (int i=1; i<GetNcat(); i++)	{
		rate[i] = GetNcat() * (y[i] - y[i-1]);
	}
	delete[] x;
	delete[] y;
   return 1.0;   
}

double GammaRateProcessVI::LogRatePrior()	{
	return -alpha;
}

// ------------------------------ Compute the Phylo-MPI distribution of GammaRateProcess -----------------------------------------------------------------------


// ---------------------------- Compute sufficient statistics of Rate Process ---------------------------------------------------------------------------------

void GammaRateProcessVI::GlobalUpdateRateSuffStat()	{
	assert(GetMyid() == 0);
	// MPI2
	// should ask the slaves to call their UpdateRateSuffStat
	// and then gather the statistics;
	int i,j,nprocs = GetNprocs(),workload = GetNcat();
	MPI_Status stat;
	MESSAGE signal = UPDATE_RATE;
	MPI_Bcast(&signal,1,MPI_INT,0,MPI_COMM_WORLD);

	for(i=0; i<workload; ++i) {
		ratesuffstatcount[i] = 0;
		ratesuffstatbeta[i] = 0.0;
	}
#ifdef BYTE_COM
	int k,l;
	double x;
	unsigned char* bvector = new unsigned char[workload*(sizeof(int)+sizeof(double))];

	for(i=1; i<nprocs; ++i) {
		MPI_Recv(bvector,workload*(sizeof(int)+sizeof(double)),MPI_UNSIGNED_CHAR,MPI_ANY_SOURCE,TAG1,MPI_COMM_WORLD,&stat);
		for(j=0; j<workload; ++j) {
			l = 0;
			for(k=sizeof(int)-1; k>=0; --k) {
				l = (l << 8) + bvector[sizeof(int)*j+k]; 
			}
			ratesuffstatcount[j] += l;
		}
		for(j=0; j<workload; ++j) {
			memcpy(&x,&bvector[sizeof(int)*workload+sizeof(double)*j],sizeof(double));
			ratesuffstatbeta[j] += x;
		}
	}
	delete[] bvector;
#else
	int ivector[workload];
	double dvector[workload];
        for(i=1; i<nprocs; ++i) {
                MPI_Recv(ivector,workload,MPI_INT,MPI_ANY_SOURCE,TAG1,MPI_COMM_WORLD,&stat);
                for(j=0; j<workload; ++j) {
                        ratesuffstatcount[j] += ivector[j];                      
                }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for(i=1; i<nprocs; ++i) {
                MPI_Recv(dvector,workload,MPI_DOUBLE,MPI_ANY_SOURCE,TAG1,MPI_COMM_WORLD,&stat);
                for(j=0; j<workload; ++j) {
                        ratesuffstatbeta[j] += dvector[j]; 
                }
        }
#endif
}

void GammaRateProcessVI::UpdateRateSuffStat()	{

	for (int i=0; i<GetNcat(); i++)	{
		ratesuffstatcount[i] = 0;
		ratesuffstatbeta[i] = 0.0;
	}
	for (int i=GetSiteMin(); i<GetSiteMax(); i++)	{
		ratesuffstatcount[alloc[i]] += GetSiteRateSuffStatCount(i);
		ratesuffstatbeta[alloc[i]] += GetSiteRateSuffStatBeta(i);
	}

}	

void GammaRateProcessVI::SlaveUpdateRateSuffStat()	{
	assert(GetMyid() > 0);

	UpdateRateSuffStat();

#ifdef BYTE_COM
	int n = 0;
	unsigned int j;
	unsigned char el_int[sizeof(int)],el_dbl[sizeof(double)];
	unsigned char* bvector = new unsigned char[GetNcat()*(sizeof(int)+sizeof(double))];

	for(int i=0; i<GetNcat(); ++i) {
		convert(el_int,ratesuffstatcount[i]);
		for(j=0; j<sizeof(int); ++j) {
			bvector[n] = el_int[j]; n++;
		}
	}
	for(int i=0; i<GetNcat(); ++i) {
		convert(el_dbl,ratesuffstatbeta[i]);
		for(j=0; j<sizeof(double); ++j) {
			bvector[n] = el_dbl[j]; n++;
		}
	}
	MPI_Send(bvector,GetNcat()*(sizeof(int)+sizeof(double)),MPI_UNSIGNED_CHAR,0,TAG1,MPI_COMM_WORLD);
	delete[] bvector;
#else
	MPI_Send(ratesuffstatcount,GetNcat(),MPI_INT,0,TAG1,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Send(ratesuffstatbeta,GetNcat(),MPI_DOUBLE,0,TAG1,MPI_COMM_WORLD);
#endif
}

// ----------------------------- Compute the derivative of variational distribution ----------------------------------------------------------------------------


// -----------------------------------------------------------------------------------------------------------------------------------------------------------
//                                             Monte Carlo Simulation
// ------------------------------------------------------------------------------------------------------------------------------------------------------------

void GammaRateProcessVI::MCQRateAlpha(int MCsamples)  {

     MCDQRateAlpha = new double[MCsamples];
     MCLogRateSuffStatProb = new double[MCsamples];
     MCLogQRate = new double[MCsamples];
     MCLogRate = new double[MCsamples];
     MCfRateAlpha = new double[MCsamples];
     MChRateAlpha = new double[MCsamples];

     GlobalUpdateRateSuffStat();

     for(int s=0; s < MCsamples; s++)  {
         MCDQRateAlpha[s] = 0;
         MCLogRateSuffStatProb[s] = 0;
         MCLogQRate[s] = 0;
         MCLogRate[s] = 0;
         MCfRateAlpha[s] = 0;
         MChRateAlpha[s] = 0;
        double* MCRate = new double[GetNcat()];
        double* xVI = new double[GetNcat()];
	double* yVI = new double[GetNcat()];
	double lgVI = rnd::GetRandom().logGamma(alpha+1.0);
	for (int k=0; k<GetNcat(); k++)	{
		xVI[k] = PointGamma((k+1.0)/GetNcat(),alpha,alpha);
	}
	for (int k=0; k<GetNcat()-1; k++)	{
		yVI[k] = IncompleteGamma(alpha*xVI[k],alpha+1,lgVI);
	}
	yVI[GetNcat()-1] = 1.0;
	MCRate[0] = GetNcat() * yVI[0];
         for(int k=1; k<GetNcat(); k++) {
             MCRate[k] = GetNcat() * (yVI[k] - yVI[k-1]);
             MCDQRateAlpha[s] += -gsl_sf_psi(alpha) + log(alpha) + 1 + log(MCRate[k]) - MCRate[k];
             MCLogRateSuffStatProb[s] += ratesuffstatcount[k] * log(MCRate[k]) - MCRate[k] * ratesuffstatbeta[k];
             MCLogQRate[s] += alpha * log(alpha) - rnd::GetRandom().logGamma(alpha) + (alpha - 1) * log(MCRate[k]) - alpha * MCRate[k];
             MCLogRate[s] += Palpha * log(Palpha) - rnd::GetRandom().logGamma(Palpha) + (Palpha -1) * log(MCRate[k]) - Palpha * MCRate[k];   
         }
         /*delete[] xVI;
         delete[] yVI;
         delete[] MCRate;
         xVI = 0;
         yVI = 0;
         MCRate = 0;*/
         MCfRateAlpha[s] = MCDQRateAlpha[s] * (MCLogRate[s] - MCLogQRate[s] + MCLogRateSuffStatProb[s]);
         MChRateAlpha[s] = MCDQRateAlpha[s];
     }
}


/*void GammaRateProcessVI::GlobalMCQRateAlpha(int MCsamples)  {
     assert(GetMyid() == 0);

     MPI_Status stat;
     MESSAGE signal = MC_RALPHA;
     MPI_Bcast(&signal,1,MPI_INT,0,MPI_COMM_WORLD);
     int itmp[1];
     itmp[0] = MCsamples;
     MPI_Bcast(itmp,1,MPI_INT,0,MPI_COMM_WORLD);
     
     int nprocs = GetNprocs();
     double ivector[MCsamples];
     double dvector[MCsamples];
         for(int i=1; i<nprocs; ++i)  {
               MPI_Recv(ivector,MCsamples,MPI_DOUBLE,MPI_ANY_SOURCE,TAG1,MPI_COMM_WORLD,&stat);
               for(int j=0; j < MCsamples; ++j) {
                   MCfRateAlpha[j] = ivector[j];    
               } 
         }
         MPI_Barrier(MPI_COMM_WORLD);
         for(int i=1; i<nprocs; ++i)  {
               MPI_Recv(dvector,MCsamples,MPI_DOUBLE,MPI_ANY_SOURCE,TAG1,MPI_COMM_WORLD,&stat);
               for(int j=0; j < MCsamples; ++j) {
                   MChRateAlpha[j] = dvector[j]; 
               }
         }
     
}

void GammaRateProcessVI::SlaveMCQRateAlpha()  {
     assert(GetMyid() > 0);

     int itmp[1];
     MPI_Bcast(itmp,1,MPI_INT,0,MPI_COMM_WORLD);
     int MCsamples = itmp[0];

     MCDQRateAlpha = new double[MCsamples];
     MCLogRateSuffStatProb = new double[MCsamples];
     MCLogQRate = new double[MCsamples];
     MCLogRate = new double[MCsamples];
     MCfRateAlpha = new double[MCsamples];
     MChRateAlpha = new double[MCsamples];

     GlobalUpdateRateSuffStat();

     for(int s=0; s < MCsamples; s++)  {
         MCDQRateAlpha[s] = 0;
         MCLogRateSuffStatProb[s] = 0;
         MCLogQRate[s] = 0;
         MCLogRate[s] = 0;
         MCfRateAlpha[s] = 0;
         MChRateAlpha[s] = 0;
         MCRate = new double[GetNcat()];
         for(int k=0; k<GetNcat(); k++) {
             MCRate[k] = rnd::GetRandom().Gamma(Ratealpha, Ratebeta);
             MCDQRateAlpha[s] += -gsl_sf_psi(Ratealpha) + log(Ratebeta) + log(MCRate[k]);
             MCLogRateSuffStatProb[s] += ratesuffstatcount[k] * log(MCRate[k]) - MCRate[k] * ratesuffstatbeta[k];
             MCLogQRate[s] += Ratealpha * log(Ratebeta) - rnd::GetRandom().logGamma(Ratealpha) + (Ratealpha - 1) * log(MCRate[k]) - Ratebeta * MCRate[k];
             MCLogRate[s] += Palpha * log(Palpha) - rnd::GetRandom().logGamma(Palpha) + (Palpha -1) * log(MCRate[k]) - Palpha * MCRate[k];   
         }
         delete[] MCRate;
         MCfRateAlpha[s] = MCDQRateAlpha[s] * (MCLogRate[s] - MCLogQRate[s] + MCLogRateSuffStatProb[s]);
         MChRateAlpha[s] = MCDQRateAlpha[s];
     }
   MPI_Send(MCfRateAlpha,MCsamples,MPI_INT,0,TAG1,MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Send(MChRateAlpha,MCsamples,MPI_INT,0,TAG1,MPI_COMM_WORLD);  
}
*/

void GammaRateProcessVI::GradQRateAlpha(int MCsamples)  {
     
     /* GlobalMCQRateAlpha(MCsamples); */
     
     MCQRateAlpha(MCsamples);
     grad_RateAlpha = 0;
     meanfRateAlpha = 0;
     meanhRateAlpha = 0;

     for(int s=0; s < MCsamples; s++) {
            meanfRateAlpha += GetMCfRateAlpha(s);
            meanhRateAlpha += GetMChRateAlpha(s);
     }
     meanfRateAlpha /= MCsamples;
     meanhRateAlpha /= MCsamples;
     varRateAlpha = 0;
     covRateAlpha = 0;
   
     for(int s=0; s < MCsamples; s++) {
           covRateAlpha += (GetMCfRateAlpha(s) - meanfRateAlpha) * (GetMChRateAlpha(s) - meanhRateAlpha);
           varRateAlpha += (GetMChRateAlpha(s) - meanhRateAlpha) * (GetMChRateAlpha(s) - meanhRateAlpha);
     }
     covRateAlpha /= MCsamples;
     varRateAlpha /= MCsamples;
     a_RateAlpha = covRateAlpha / varRateAlpha;

     for(int s=0; s < MCsamples; s++)  {
           grad_RateAlpha += (GetMCfRateAlpha(s) - a_RateAlpha * GetMChRateAlpha(s));
     }
     grad_RateAlpha /= MCsamples;

     delete[] MCDQRateAlpha;
     delete[] MCLogRateSuffStatProb;
     delete[] MCLogQRate;
     delete[] MCLogRate;
     delete[] MCfRateAlpha;
     delete[] MChRateAlpha;
}

// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                            Stochastic Optimization for Variational Parameters
// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

double GammaRateProcessVI::MoveRateAlpha(int MCsamples)  {
     // double VIalpha = alpha;
     GradQRateAlpha(MCsamples);
     GRateAlpha += Getgrad_RateAlpha() * Getgrad_RateAlpha();
     alpha += (0.01 * Getgrad_RateAlpha()) / sqrt(GRateAlpha + 1e-8);
  return alpha;
}

double GammaRateProcessVI::Move(int MCsamples) {
     GlobalUpdateSiteRateSuffStat();
     MoveRateAlpha(MCsamples);
     UpdateDiscreteCategories();
     return 1.0;
}


