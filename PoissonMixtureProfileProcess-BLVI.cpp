
/********************



**********************/


#include "PoissonMixtureProfileProcessVI.h"
#include "Random.h"
#include <gsl/gsl_sf_psi.h>

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
//	* PoissonMixtureProfileProcess
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------


void PoissonMixtureProfileProcessVI::Create(int innsite, int indim)	{
       double Gdirweight[GetDim()];
       if (! profilesuffstatcount)	{
		PoissonProfileProcess::Create(innsite,indim);
		MixtureProfileProcess::Create(innsite,indim);
		profilesuffstatcount  = new int*[GetNmodeMax()];
		for (int i=0; i<GetNmodeMax(); i++)	{
			profilesuffstatcount[i] = new int[GetDim()];
		}
		// SampleProfile();
	}
}

void PoissonMixtureProfileProcessVI::Delete() {
	if (profilesuffstatcount)	{
		for (int i=0; i<GetNmodeMax(); i++)	{
			delete[] profilesuffstatcount[i];
		}
		delete[] profilesuffstatcount;
		profilesuffstatcount = 0;
		PoissonProfileProcess::Delete();
		MixtureProfileProcess::Delete();
	}
}

void PoissonMixtureProfileProcessVI::UpdateModeProfileSuffStat()	{
	for (int i=0; i<GetNcomponent(); i++)	{
		for (int k=0; k<GetDim(); k++)	{
			profilesuffstatcount[i][k] = 0;
		}
	}
	for (int i=0; i<GetNsite(); i++)	{
		const int* count = GetSiteProfileSuffStatCount(i);
		int cat = alloc[i];
		for (int k=0; k<GetDim(); k++)	{
			profilesuffstatcount[cat][k] += count[k];
		}
	}
}

double PoissonMixtureProfileProcessVI::DiffLogSampling(int cat, int site)	{

	const int* nsub = GetSiteProfileSuffStatCount(site);
	int* catnsub = profilesuffstatcount[cat];
	int totalsub = 0;
	double priorweight = 0;
	int grandtotal = 0;
	for (int k=0; k<GetDim(); k++)	{
		totalsub += nsub[k];
		priorweight += dirweight[k];
		grandtotal += catnsub[k];
	}
	
	double total = 0;
	for (int j=0; j< totalsub; j++)	{
		total -= log(priorweight + grandtotal + j);
	}
	for (int k=0; k<GetDim(); k++)	{
		for (int j=0; j< nsub[k]; j++)	{
			total += log(dirweight[k] + catnsub[k] + j);
		}
	}
	return total;
}

double PoissonMixtureProfileProcessVI::LogStatProb(int site, int cat)	{
	const int* nsub = GetSiteProfileSuffStatCount(site);
	double total = 0;
	for (int k=0; k<GetDim(); k++)	{
		total += nsub[k] * log(profile[cat][k]);
	}
	return total;
}

void PoissonMixtureProfileProcessVI::RemoveSite(int site, int cat)	{
	occupancy[cat] --;
	if (activesuffstat)	{
		const int* nsub = GetSiteProfileSuffStatCount(site);
		int* catnsub = profilesuffstatcount[cat];
		for (int k=0; k<GetDim(); k++)	{
			catnsub[k] -= nsub[k];
		}
	}
}

void PoissonMixtureProfileProcessVI::AddSite(int site, int cat)	{
	alloc[site] = cat;
	occupancy[cat] ++;
	UpdateZip(site);
	if (activesuffstat)	{
		const int* nsub = GetSiteProfileSuffStatCount(site);
		int* catnsub = profilesuffstatcount[cat];
		for (int k=0; k<GetDim(); k++)	{
			catnsub[k] += nsub[k];
		}
	}
}

void PoissonMixtureProfileProcessVI::SwapComponents(int cat1, int cat2)	{

	MixtureProfileProcess::SwapComponents(cat1,cat2);
	for (int k=0; k<GetDim(); k++)	{
		int tmp = profilesuffstatcount[cat1][k];
		profilesuffstatcount[cat1][k] = profilesuffstatcount[cat2][k];
		profilesuffstatcount[cat2][k] = tmp;
	}
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                     Variational distribution 
// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

double PoissonMixtureProfileProcessVI::ProfileSuffStatLogProb(int cat)	{
	double total = 0;
	double priorweight = 0;
	double postweight = 0;
	for (int k=0; k<GetDim(); k++)	{
		total += rnd::GetRandom().logGamma(dirweight[k] + profilesuffstatcount[cat][k]) - rnd::GetRandom().logGamma(dirweight[k]);
		priorweight += dirweight[k];
		postweight += dirweight[k] + profilesuffstatcount[cat][k];
	}
	total += rnd::GetRandom().logGamma(priorweight) - rnd::GetRandom().logGamma(postweight);
	return total;
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                      Monte Carlo Simulation
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void PoissonMixtureProfileProcessVI::MCQdirweight(int MCsamples)  {
      
      UpdateOccupancyNumbers();
      UpdateModeProfileSuffStat(); 

      MCDQdirweight = new double[MCsamples];
      MCLogStatProb = new double[MCsamples];
      MCLogQdirweight = new double[MCsamples];
      MCLogdirweight = new double[MCsamples];
      MCfdirweight = new double[MCsamples];
      MChdirweight = new double[MCsamples];

      Pdirweight[GetDim()];
      MCProfile = new double**[MCsamples];
      
      for(int s=0; s<MCsamples; s++)  {
           MCDQdirweight[s] = 0;
           MCLogStatProb[s] = 0;
           MCLogQdirweight[s] = 0;
           MCLogdirweight[s] = 0;
           MCfdirweight[s] = 0;
           MChdirweight[s] = 0;
           double* MCDQdirweighti = new double[GetNcomponent()];
           double* MCLogStatProbi = new double[GetNcomponent()];
           double* MCLogQdirweighti = new double[GetNcomponent()];
           double* MCLogdirweighti = new double[GetNcomponent()];
           MCProfile[s] = new double*[GetNcomponent()];
                 for(int i=0; i<GetNcomponent(); i++)  {
                         MCDQdirweighti[i] = 0;
                         MCLogStatProbi[i] = 0;
                         MCLogQdirweighti[i] = 0;
                         MCLogdirweighti[i] = 0;
                         MCProfile[s][i] = new double[GetDim()];
                         double totalprofile = 0;
                                  for(int k=0; k<GetDim(); k++)  {
                                            MCProfile[s][i][k] = rnd::GetRandom().sGamma(dirweight[k]);
                                            if (MCProfile[s][i][k] < stateps) {
                                                MCProfile[s][i][k] = stateps;
                                            }
                                            totalprofile += MCProfile[s][i][k];
                                  } 
                                  double totalweight = 0;
                                  double Ptotalweight = 0;
                                  for(int k=0; k<GetDim(); k++)  {
                                          Pdirweight[k] = 1;
                                          MCProfile[s][i][k] /= totalprofile;
                                          MCDQdirweighti[i] += log(MCProfile[s][i][k]) - gsl_sf_psi(dirweight[k]);
                                          totalweight += dirweight[k]; 
                                          MCLogQdirweighti[i] += (dirweight[k] - 1) * log(MCProfile[s][i][k]) - rnd::GetRandom().logGamma(dirweight[k]);
                                          MCLogdirweighti[i] += (Pdirweight[k] -1 ) * log(MCProfile[s][i][k]) - rnd::GetRandom().logGamma(Pdirweight[k]);
                                          Ptotalweight += Pdirweight[k];
                                          MCLogStatProbi[i] += profilesuffstatcount[i][k] * log(MCProfile[s][i][k]);
                                  }
                                  MCDQdirweighti[i] += gsl_sf_psi(totalweight);
                                  MCLogQdirweighti[i] += rnd::GetRandom().logGamma(totalweight);
                                  MCLogdirweighti[i] += rnd::GetRandom().logGamma(Ptotalweight);
                    MCDQdirweight[s] += MCDQdirweighti[i];
                    MCLogQdirweight[s] += MCLogQdirweighti[i];
                    MCLogdirweight[s] += MCLogdirweighti[i];
                    MCLogStatProb[s] += MCLogStatProbi[i];
                 }
          MCfdirweight[s] = MCDQdirweight[s] * (MCLogdirweight[s] - MCLogQdirweight[s] + MCLogStatProb[s]);
          MChdirweight[s] = MCDQdirweight[s];
          delete[] MCDQdirweighti;
          delete[] MCLogQdirweighti;
          delete[] MCLogdirweighti;
          delete[] MCLogStatProbi;
      }
   delete[] MCDQdirweight;
   delete[] MCLogQdirweight;
   delete[] MCLogdirweight;
   delete[] MCLogStatProb;
}

void PoissonMixtureProfileProcessVI::Graddirweight(int MCsamples)  {
     MCQdirweight(MCsamples);
     
     grad_dirweight = 0;
     meanfdirweight = 0;
     meanhdirweight = 0; 
   
     for (int s=0; s<MCsamples; s++)  {
           meanfdirweight += GetMCfdirweight(s);
           meanhdirweight += GetMChdirweight(s);
     }
     meanfdirweight /= MCsamples;
     meanhdirweight /= MCsamples;
     vardirweight = 0;
     covdirweight = 0;
   
     for (int s=0; s<MCsamples; s++)  {
           covdirweight += (GetMCfdirweight(s) - meanfdirweight) * (GetMChdirweight(s) - meanhdirweight);
           vardirweight += (GetMChdirweight(s) - meanhdirweight) * (GetMChdirweight(s) - meanhdirweight);
     }
     covdirweight /= MCsamples;
     vardirweight /= MCsamples;
     a_dirweight = covdirweight / vardirweight;

     for (int s=0; s<MCsamples; s++)  {
           grad_dirweight += (GetMCfdirweight(s) - a_dirweight * GetMChdirweight(s));
     }
     grad_dirweight /= MCsamples;
    
     for (int s=0; s<MCsamples; s++)  {
              for(int i; i<GetNcomponent(); i++) {
                   delete[] MCProfile[s][i];
              }
          delete[] MCProfile[s]; 
     }
   delete[] MCProfile;  
   delete[] MCfdirweight;
   delete[] MChdirweight;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                            Stochastic Optimization for Variational Parameter
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

double PoissonMixtureProfileProcessVI::MoveDirWeight(int MCsamples)  {

       for (int k=0; k<GetDim(); k++)  {
            Graddirweight(MCsamples);
            Gdirweight[k] += Getgrad_dirweight() * Getgrad_dirweight();
            dirweight[k] += (0.01 * Getgrad_dirweight()) / sqrt(Gdirweight[k] + 1e-8);
       }
   return 1.0;
}

double PoissonMixtureProfileProcessVI::MoveProfile()	{
	for (int i=0; i<GetNcomponent(); i++)	{
		MoveProfile(i);
	}
	return 1;
}

double PoissonMixtureProfileProcessVI::MoveProfile(int cat)	{
	double total = 0;
	for (int k=0; k<GetDim(); k++)	{
		profile[cat][k] = rnd::GetRandom().sGamma(dirweight[k]);
		if (profile[cat][k] < stateps)	{
			profile[cat][k] = stateps;
		}
		total += profile[cat][k];
	}
	for (int k=0; k<GetDim(); k++)	{
		profile[cat][k] /= total;
	}
	return 1;
}

