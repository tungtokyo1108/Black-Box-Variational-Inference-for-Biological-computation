
/********************


**********************/


#include "SBDPProfileProcessVI.h"
#include "Random.h"
#include <gsl/gsl_sf_psi.h>

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
//	* SBDPProfileProcessVI
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------


void SBDPProfileProcessVI::Create(int innsite, int indim)	{
        
        kappa_alpha = 1.0;
	if (! V)	{
		DPProfileProcessVI::Create(innsite,indim);
		V = new double[GetNmodeMax()];
                V_P = new double[GetNmodeMax()];
		weight = new double[GetNmodeMax()];
                weight_P = new double[GetNmodeMax()];
	}
        GVAlpha = rnd::GetRandom().Uniform();
        GVBeta = rnd::GetRandom().Uniform();
}

void SBDPProfileProcessVI::Delete()	{

	if (V)	{
		delete[] V;
		delete[] weight;
                delete[] V_P;
                delete[] weight_P;
		DPProfileProcessVI::Delete();
	}
}

void SBDPProfileProcessVI::SampleAlloc()	{

	for (int k=0; k<GetNmodeMax(); k++)	{
		CreateComponent(k);
	}
	Ncomponent = GetNmodeMax();

	SampleWeights();
	for (int i=0; i<GetNsite(); i++)	{
		double U = rnd::GetRandom().Uniform();
		double total = weight[0];
		int k = 0;
		while ((k<GetNmodeMax()) && (total < U))	{
			k++;
			total += weight[k];
		}
		if (k == GetNmodeMax())	{
			cerr << "error in SBDPProfileProcess::SampleAlloc: overflow\n";
			exit(1);
		}
		AddSite(i,k);
	}
}


void SBDPProfileProcessVI::DrawProfileFromPrior()	{

	if (! GetMyid())	{
		cerr << "error: in master DrawProfileFromPrior\n";
		exit(1);
	}

	for (int i=GetSiteMin(); i<GetSiteMax(); i++)	{
		RemoveSite(i,alloc[i]);
		int choose = rnd::GetRandom().FiniteDiscrete(GetNcomponent(),weight);
		AddSite(i,choose);
	}
}

void SBDPProfileProcessVI::IncrementalSampleAlloc()	{

	kappa = 0.1;

	for (int i=0; i<GetNsite(); i++)	{
		RemoveSite(i,alloc[i]);
	}

	AddSite(0,0);
	Ncomponent = 1;
	
	for (int i=0; i<GetNsite(); i++)	{

		int K = Ncomponent + 1;
		if (K > GetNmodeMax())	{
			K--;
		}
		double* p = new double[K];
		double total = 0;
		double max = 0;
		for (int k=0; k<K; k++)	{
			double w = occupancy[k];
			if (! w)	{
				w = kappa;
			}
			double tmp = log(w) * LogProxy(i,k);
			if ((!k) || (max < tmp))	{
				max = tmp;
			}
			p[k] = tmp;
		}
		for (int k=0; k<K; k++)	{
			double tmp = exp(p[k] - max);
			total += tmp;
			p[k] = total;
		}
		double q = total * rnd::GetRandom().Uniform();
		int k = 0;
		while ((k<K) && (q > p[k])) k++;
		if (k == K)	{
			cerr << "error in draw dp mode: overflow\n";
			exit(1);
		}
		if (k==Ncomponent)	{
			if (Ncomponent <= GetNmodeMax())	{
				Ncomponent++;
			}
		}
		AddSite(i,k);
		delete[] p;
	}

	Ncomponent = GetNmodeMax();
	ResampleWeights();
	cerr << "init incremental ok\n";
}

double SBDPProfileProcessVI::LogStatPrior()	{

	UpdateOccupancyNumbers();
	double total = 0;
	for (int i=0; i<GetNcomponent(); i++)	{
		if (occupancy[i])	{
			total += DPProfileProcessVI::LogStatPrior(i);
		}
	}
	return total;
}

void SBDPProfileProcessVI::SwapComponents(int cat1, int cat2)	{

	MixtureProfileProcess::SwapComponents(cat1,cat2);
	double tempv = V[cat1];
	V[cat1] = V[cat2];
	V[cat2] = tempv;
	double tempw = weight[cat1];
	weight[cat1] = weight[cat2];
	weight[cat2] = tempw;
}

void SBDPProfileProcessVI::SampleWeights()	{
        
        
	double cumulProduct = 1.0;
	double totweight = 0;
	double v, x, y;
	for (int k=0; k<GetNcomponent(); k++)	{
		x = rnd::GetRandom().sGamma(kappa_alpha);
		y = rnd::GetRandom().sGamma(kappa);
		v = x / (x+y);
		V[k] = v;
		if (k == GetNcomponent() - 1)	{
			V[k] = 1;
			v = 1;
		}
		weight[k] = v * cumulProduct;
		cumulProduct *= (1 - v);	
		totweight += weight[k];
	}
}


void SBDPProfileProcessVI::ResampleWeights()	{

	UpdateOccupancyNumbers();
	// ???
	int remainingOcc = GetNsite();
	double cumulProduct = 1.0;
	double totweight = 0;
	double v, x, y;
	for (int k=0; k<GetNcomponent(); k++)	{
		remainingOcc -= occupancy[k];
		x = rnd::GetRandom().sGamma(kappa_alpha);
		y = rnd::GetRandom().sGamma(kappa);
		v = x / (x+y);
		V[k] = v;
		if (k == GetNcomponent() - 1)	{
			double tmp = cumulProduct * (1 - v);
			if (maxweighterror < tmp)	{
			    maxweighterror = tmp;
			}
			V[k] = 1;
			v = 1;
		}
                weight[k] = v * cumulProduct;
                cumulProduct *= (1 - v);
                totweight += weight[k];
		
	}
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                   Monte Carlo Simulation 
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void SBDPProfileProcessVI::MCQVAlpha(int MCsamples)  {

    MCDQVAlpha = new double[MCsamples];
    MCLogIntegratedAllocProb = new double[MCsamples];
    MCLogQV = new double[MCsamples];
    MCLogV = new double[MCsamples];
    MCfVAlpha = new double[MCsamples];
    MChVAlpha = new double[MCsamples];

    int remainingOcc = GetNsite();
    double v, x, y;

    MCV = new double*[MCsamples];
    
    Pkappa_alpha = 1.0;
    Pkappa = 0.1;

    for(int s=0; s<MCsamples; s++)  {
         MCDQVAlpha[s] = 0;
         MCLogIntegratedAllocProb[s] = 0;
         MCLogQV[s] = 0;
         MCLogV[s] = 0;
         MCfVAlpha[s] = 0;
         MChVAlpha[s] = 0;
         MCV[s] = new double[GetNcomponent()]; 
         for (int k=0; k<GetNcomponent(); k++)  {
                  remainingOcc -= occupancy[k];
                  x = rnd::GetRandom().sGamma(kappa_alpha);
	          y = rnd::GetRandom().sGamma(kappa);
	          v = x / (x+y);
	          MCV[s][k] = v;
                  MCDQVAlpha[s] += gsl_sf_psi(kappa_alpha + kappa) - gsl_sf_psi(kappa_alpha) + log(MCV[s][k]);
                  MCLogIntegratedAllocProb[s] += remainingOcc * log(1 - MCV[s][k]);
                  MCLogIntegratedAllocProb[s] += occupancy[k] * log(MCV[s][k]);
                  MCLogQV[s] += rnd::GetRandom().logGamma(kappa_alpha + kappa) - rnd::GetRandom().logGamma(kappa_alpha) - rnd::GetRandom().logGamma(kappa); 
                  MCLogQV[s] += (kappa_alpha - 1) * rnd::GetRandom().logGamma(MCV[s][k]) + (kappa - 1) * log(1 - MCV[s][k]);
                  MCLogV[s] += rnd::GetRandom().logGamma(Pkappa_alpha + Pkappa) - rnd::GetRandom().logGamma(Pkappa_alpha) -rnd::GetRandom().logGamma(Pkappa);
                  MCLogV[s] += (Pkappa_alpha -1) * rnd::GetRandom().logGamma(MCV[s][k]) + (kappa -1) * log(1 - MCV[s][k]);
         }  
    MCfVAlpha[s] = MCDQVAlpha[s] * (MCLogV[s] - MCLogQV[s] + MCLogIntegratedAllocProb[s]);
    MChVAlpha[s] = MCDQVAlpha[s];
    }
}


void SBDPProfileProcessVI::GradVAlpha(int MCsamples)  {

    MCQVAlpha(MCsamples);

    grad_VAlpha = 0;
    meanfVAlpha = 0;
    meanhVAlpha = 0;

    for (int s=0; s<MCsamples; s++) {
          meanfVAlpha += GetMCfVAlpha(s);
          meanhVAlpha += GetMChVAlpha(s);         
    }
    meanfVAlpha /= MCsamples;
    meanhVAlpha /= MCsamples;
    varVAlpha = 0;
    covVAlpha = 0;

    for (int s=0; s<MCsamples; s++) {
          covVAlpha += (GetMCfVAlpha(s) - meanfVAlpha) * (GetMChVAlpha(s) - meanhVAlpha);
          varVAlpha += (GetMChVAlpha(s) - meanhVAlpha) * (GetMChVAlpha(s) - meanhVAlpha);
    }
    covVAlpha /= MCsamples;
    varVAlpha /= MCsamples;
    a_VAlpha = covVAlpha / varVAlpha;

    for (int s=0; s<MCsamples; s++)  {
          grad_VAlpha += (GetMCfVAlpha(s) - a_VAlpha * GetMChVAlpha(s));
    }
    grad_VAlpha /= MCsamples;
 
    for (int s=0; s<MCsamples; s++) {
         delete[] MCV[s];
    }
 delete[] MCV;
 delete[] MCDQVAlpha;
 delete[] MCLogIntegratedAllocProb;
 delete[] MCLogQV;
 delete[] MCLogV;
 delete[] MCfVAlpha;
 delete[] MChVAlpha;

}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void SBDPProfileProcessVI::MCQVBeta(int MCsamples)  {
    
    MCDQVBeta = new double[MCsamples];
    MCLogIntegratedAllocProb = new double[MCsamples];
    MCLogQV = new double[MCsamples];
    MCLogV = new double[MCsamples];
    MCfVBeta = new double[MCsamples];
    MChVBeta = new double[MCsamples];
    int remainingOcc = GetNsite();
    double v, x, y;

    MCV = new double*[MCsamples];
    
    Pkappa_alpha = 1.0;
    Pkappa = 0.1;

    for(int s=0; s<MCsamples; s++)  {
         MCDQVBeta[s] = 0;
         MCLogIntegratedAllocProb[s] = 0;
         MCLogQV[s] = 0;
         MCLogV[s] = 0;
         MCfVBeta[s] = 0;
         MChVBeta[s] = 0;
         MCV[s] = new double[GetNcomponent()]; 
         for (int k=0; k<GetNcomponent(); k++)  {
                  remainingOcc -= occupancy[k];
                  x = rnd::GetRandom().sGamma(kappa_alpha);
	          y = rnd::GetRandom().sGamma(kappa);
	          v = x / (x+y);
	          MCV[s][k] = v;
                  MCDQVBeta[s] += gsl_sf_psi(kappa_alpha + kappa) - gsl_sf_psi(kappa) + log(1-MCV[s][k]);
                  MCLogIntegratedAllocProb[s] += remainingOcc * log(1 - MCV[s][k]);
                  MCLogIntegratedAllocProb[s] += occupancy[k] * log(MCV[s][k]);
                  MCLogQV[s] += rnd::GetRandom().logGamma(kappa_alpha + kappa) - rnd::GetRandom().logGamma(kappa_alpha) - rnd::GetRandom().logGamma(kappa); 
                  MCLogQV[s] += (kappa_alpha - 1) * rnd::GetRandom().logGamma(MCV[s][k]) + (kappa - 1) * log(1 - MCV[s][k]);
                  MCLogV[s] += rnd::GetRandom().logGamma(Pkappa_alpha + Pkappa) - rnd::GetRandom().logGamma(Pkappa_alpha) -rnd::GetRandom().logGamma(Pkappa);
                  MCLogV[s] += (Pkappa_alpha -1) * rnd::GetRandom().logGamma(MCV[s][k]) + (kappa -1) * log(1 - MCV[s][k]);
         }  
    MCfVBeta[s] = MCDQVBeta[s] * (MCLogV[s] - MCLogQV[s] + MCLogIntegratedAllocProb[s]);
    MChVBeta[s] = MCDQVBeta[s];
    }
}

void SBDPProfileProcessVI::GradVBeta(int MCsamples) {
    
    MCQVBeta(MCsamples);

    grad_VBeta = 0;
    meanfVBeta = 0;
    meanhVBeta = 0;

    for (int s=0; s<MCsamples; s++) {
          meanfVBeta += GetMCfVBeta(s);
          meanhVBeta += GetMChVBeta(s);         
    }
    meanfVBeta /= MCsamples;
    meanhVBeta /= MCsamples;
    varVBeta = 0;
    covVBeta = 0;

    for (int s=0; s<MCsamples; s++) {
          covVBeta += (GetMCfVBeta(s) - meanfVBeta) * (GetMChVBeta(s) - meanhVBeta);
          varVBeta += (GetMChVBeta(s) - meanhVBeta) * (GetMChVBeta(s) - meanhVBeta);
    }
    covVBeta /= MCsamples;
    varVBeta /= MCsamples;
    a_VBeta = covVBeta / varVBeta;

    for (int s=0; s<MCsamples; s++)  {
          grad_VBeta += (GetMCfVBeta(s) - a_VBeta * GetMChVBeta(s));
    }
    grad_VBeta /= MCsamples;
 
    for (int s=0; s<MCsamples; s++) {
         delete[] MCV[s];
    }
 delete[] MCV;
 delete[] MCDQVBeta;
 delete[] MCLogIntegratedAllocProb;
 delete[] MCLogQV;
 delete[] MCLogV;
 delete[] MCfVBeta;
 delete[] MChVBeta;

}


//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                    Stochastic Optimization for Variational Parameter 
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

double SBDPProfileProcessVI::MoveVAlpha(int MCsamples)  {
       GradVAlpha(MCsamples);
       GVAlpha += Getgrad_VAlpha() * Getgrad_VAlpha();
       kappa_alpha += (0.01 * Getgrad_VAlpha()) / sqrt(GVAlpha + 1e-8);
    return kappa_alpha;
}

double SBDPProfileProcessVI::MoveVBeta(int MCsamples)   {
       GradVBeta(MCsamples);
       GVBeta += Getgrad_VBeta() * Getgrad_VBeta();
       kappa += (0.01 * Getgrad_VBeta()) / sqrt(GVBeta + 1e-8);
    return kappa; 
}

double SBDPProfileProcessVI::MoveKappa(int MCsamples) {
       MoveVAlpha(MCsamples);
       MoveVBeta(MCsamples);
 
    return 1.0;   
}

