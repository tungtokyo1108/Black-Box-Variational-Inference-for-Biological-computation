
/********************



**********************/


#include "GammaBranchProcessVI.h"

#include <iostream>
#include <cassert>
#include "Parallel.h"
#include <string.h>

void GammaBranchProcessVI::ToStream(ostream& os)	{

	SetNamesFromLengths();
	tree->ToStream(os);
	os << branchalpha << '\n';
	os << branchbeta << '\n';
	/*
	for (int j=0; j<GetNbranch(); j++)	{
		os << blarray[j] << '\t';
	}
	os << '\n';
	*/
}

void GammaBranchProcessVI::ToStreamWithLengths(ostream& os, const Link* from)	{

	if (from->isLeaf())	{
		os << from->GetNode()->GetName();
	}
	else	{
		os << "(";
		for (const Link* link=from->Next(); link!=from; link=link->Next())	{
			ToStreamWithLengths(os, link->Out());
			if (link->Next() != from)	{
				os << ",";
			}
		}
		os << ")";
	}
	if (! from->isRoot())	{
		os << ":" << blarray[from->GetBranch()->GetIndex()];
	}
}


void GammaBranchProcessVI::FromStream(istream& is)	{

	tree->ReadFromStream(is);
	tree->RegisterWith(tree->GetTaxonSet());
	SetLengthsFromNames();
	branchalpha = -1;
	branchbeta = -1;
	is >> branchalpha;
	is >> branchbeta;
	/*
	for (int j=0; j<GetNbranch(); j++)	{
		is >> blarray[j];
	}
	*/
}

// -------------- Compute the variational distribution of GammaBranchProcess ------------------------------------------------------------------------------------ 
	
double GammaBranchProcessVI::LogBranchLengthPrior(const Branch* branch)	{
	int index = branch->GetIndex();
	return branchalpha * log(branchbeta) - rnd::GetRandom().logGamma(branchalpha) + (branchalpha-1) * log(blarray[index]) - branchbeta * blarray[index];
}

void GammaBranchProcessVI::SampleLength(const Branch* branch)	{
	int index = branch->GetIndex();
	blarray[index] = rnd::GetRandom().Gamma(branchalpha,branchbeta);
}
	
void GammaBranchProcessVI::SampleLength()	{
	cerr << "sample length\n";
	exit(1);
	branchalpha = rnd::GetRandom().sExpo();
	branchbeta = rnd::GetRandom().sExpo();
	blarray[0] = 0;
	// SampleLength();
}

/*void GammaBranchProcessVI::SampleLength()  {
       branchalpha = 1;
       branchbeta = 10;
       for (int i=1; i<GetNbranch(); i++)  {
		blarray[i] = rnd::GetRandom().Gamma(branchalpha, branchbeta);
       }
}*/

double GammaBranchProcessVI::MoveLength()	{
	for (int i=1; i<GetNbranch(); i++)	{
		blarray[i] = rnd::GetRandom().Gamma(branchalpha, branchbeta);
	}
        return 1.0;
}

// -------------Compute the Phylo-MPI distribution of GammaBranchProcess --------------------------------------------------------------------------------------------------- 

double GammaBranchProcessVI::LogPBranchLengthPrior(const Branch* branch){
        int index = branch->GetIndex();
        return Pbranchalpha * log(Pbranchbeta) - rnd::GetRandom().logGamma(Pbranchalpha) + (Pbranchalpha -1) * log(blarray[index]) - Pbranchbeta * blarray[index];
}

void GammaBranchProcessVI::PSampleLength(const Branch* branch) {
        int index = branch->GetIndex();
        blarray[index] = rnd::GetRandom().Gamma(Pbranchalpha,Pbranchbeta);
}

void GammaBranchProcessVI::PSampleLength() {
        cerr << "variational sample length\n";
        exit(1);
        Pbranchalpha = rnd::GetRandom().sExpo();
        Pbranchbeta = rnd::GetRandom().sExpo();
        blarray[0] = 0;
}

// ------------- Compute the derivative of variational distribution ------------------------------------------------------------------------------------------------------- 

double GammaBranchProcessVI::DQBranchAlpha(const Branch* branch) {
         int index = branch->GetIndex();
         return dqbranchalpha = -gsl_sf_psi(branchalpha) + log(branchbeta) + log(blarray[index]);
}
 
double GammaBranchProcessVI::DQBranchBeta(const Branch* branch) {
         int index = branch->GetIndex();
         return branchalpha/branchbeta - blarray[index];
}
 
// ---------------------------------------------------------------------------------------------------------------------------
//                                  Monte Carlo simulation 
//----------------------------------------------------------------------------------------------------------------------------

void GammaBranchProcessVI::MCQbranchalpha(int MCsamples) {
   
   MCDQbranchalpha = new double[MCsamples];
   MCLogLikelihoodBranch = new double[MCsamples];
   MCLogQBranchLengthPrior = new double[MCsamples];
   MCLogBranchLengthPrior = new double[MCsamples];
   MCfbranchalpha = new double[MCsamples];
   MChbranchalpha = new double[MCsamples];
   
   GlobalUpdateBranchLengthSuffStat();
   
   for (int s=0; s < MCsamples; s++) {
        MCDQbranchalpha[s] = 0;
        MCLogLikelihoodBranch[s] = 0;
        MCLogQBranchLengthPrior[s] = 0;
        MCLogBranchLengthPrior[s] = 0;
        MCfbranchalpha[s] = 0;
        MChbranchalpha[s] = 0;
        MCblarray = new double[GetNbranch()]; 
             for (int i=1; i<GetNbranch(); i++)	{
             MCblarray[i] = rnd::GetRandom().Gamma(branchalpha,branchbeta);
             MCDQbranchalpha[s] += -gsl_sf_psi(branchalpha) + log(branchbeta) + log(MCblarray[i]);
             MCLogLikelihoodBranch[s] -= MCblarray[i] * GetBranchLengthSuffStatBeta(i);
             MCLogLikelihoodBranch[s] += log(MCblarray[i]) * GetBranchLengthSuffStatCount(i);
             MCLogBranchLengthPrior[s] += Pbranchalpha * log(Pbranchbeta) - rnd::GetRandom().logGamma(Pbranchalpha) + (Pbranchalpha - 1) * log(MCblarray[i]) - Pbranchbeta * MCblarray[i];
             MCLogQBranchLengthPrior[s] += branchalpha * log(branchbeta) - rnd::GetRandom().logGamma(branchalpha) + (branchalpha - 1) * log(MCblarray[i]) - branchbeta * MCblarray[i];
             }
        delete[] MCblarray;
        MCfbranchalpha[s] = MCDQbranchalpha[s] * (MCLogBranchLengthPrior[s] - MCLogQBranchLengthPrior[s] + MCLogLikelihoodBranch[s]);
        MChbranchalpha[s] = MCDQbranchalpha[s];         
    }
}

/*void GammaBranchProcessVI::GlobalMCQbranchalpha(int MCsamples) {
     assert(GetMyid() == 0);

     MPI_Status stat;
     MESSAGE signal = MC_BALPHA;
     MPI_Bcast(&signal,1,MPI_INT,0,MPI_COMM_WORLD);
     int itmp[1];
     itmp[0] = MCsamples;
     MPI_Bcast(itmp,1,MPI_INT,0,MPI_COMM_WORLD); 
     
     int nprocs = GetNprocs();
     double ivector[MCsamples];
     double dvector[MCsamples];
        for(int i=1; i<nprocs; ++i) {
              MPI_Recv(ivector,MCsamples,MPI_DOUBLE,MPI_ANY_SOURCE,TAG1,MPI_COMM_WORLD,&stat);
              for(int j=0; j < MCsamples; ++j) {
                  MCfbranchalpha[j] = ivector[j];
              }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i=1; i<nprocs; ++i)  {
              MPI_Recv(dvector,MCsamples,MPI_DOUBLE,MPI_ANY_SOURCE,TAG1,MPI_COMM_WORLD,&stat);
              for(int j=0; j < MCsamples; ++j)  {
                  MChbranchalpha[j] = dvector[j];
              }
        }
}

void GammaBranchProcessVI::SlaveMCQbranchalpha()  { 
   assert(GetMyid() > 0);
   
   int itmp[1];
   MPI_Bcast(itmp,1,MPI_INT,0,MPI_COMM_WORLD);
   int MCsamples = itmp[0];

   MCDQbranchalpha = new double[MCsamples];
   MCLogLikelihoodBranch = new double[MCsamples];
   MCLogQBranchLengthPrior = new double[MCsamples];
   MCLogBranchLengthPrior = new double[MCsamples];
   MCfbranchalpha = new double[MCsamples];
   MChbranchalpha = new double[MCsamples];
   
   GlobalUpdateBranchLengthSuffStat();
   
   for (int s=0; s < MCsamples; s++) {
        MCDQbranchalpha[s] = 0;
        MCLogLikelihoodBranch[s] = 0;
        MCLogQBranchLengthPrior[s] = 0;
        MCLogBranchLengthPrior[s] = 0;
        MCfbranchalpha[s] = 0;
        MChbranchalpha[s] = 0;
        MCblarray = new double[GetNbranch()]; 
             for (int i=1; i<GetNbranch(); i++)	{
             MCblarray[i] = rnd::GetRandom().Gamma(branchalpha,branchbeta);
             MCDQbranchalpha[s] += -gsl_sf_psi(branchalpha) + log(branchbeta) + log(MCblarray[i]);
             MCLogLikelihoodBranch[s] -= MCblarray[i] * GetBranchLengthSuffStatBeta(i);
             MCLogLikelihoodBranch[s] += log(MCblarray[i]) * GetBranchLengthSuffStatCount(i);
             MCLogBranchLengthPrior[s] += Pbranchalpha * log(Pbranchbeta) - rnd::GetRandom().logGamma(Pbranchalpha) + (Pbranchalpha - 1) * log(MCblarray[i]) - Pbranchbeta * MCblarray[i];
             MCLogQBranchLengthPrior[s] += branchalpha * log(branchbeta) - rnd::GetRandom().logGamma(branchalpha) + (branchalpha - 1) * log(MCblarray[i]) - branchbeta * MCblarray[i];
             }
        delete[] MCblarray;
        MCfbranchalpha[s] = MCDQbranchalpha[s] * (MCLogBranchLengthPrior[s] - MCLogQBranchLengthPrior[s] + MCLogLikelihoodBranch[s]);
        MChbranchalpha[s] = MCDQbranchalpha[s];         
    }
  MPI_Send(MCfbranchalpha,MCsamples,MPI_INT,0,TAG1,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);   
  MPI_Send(MChbranchalpha,MCsamples,MPI_INT,0,TAG1,MPI_COMM_WORLD);
}
*/

void GammaBranchProcessVI::GradQbranchalpha(int MCsamples) {
    
    /* GlobalMCQbranchalpha(MCsamples); */
    
    MCQbranchalpha(MCsamples);
    grad_branchalpha = 0;
    meanfbranchalpha = 0; 
    meanhbranchalpha = 0;  

   for (int s=0; s < MCsamples; s++) {
            meanfbranchalpha += GetMCfbranchalpha(s);
            meanhbranchalpha += GetMChbranchalpha(s);     
   }
   meanfbranchalpha /= MCsamples;
   meanhbranchalpha /= MCsamples;
   varbranchalpha = 0; 
   covbranchalpha = 0; 

   for (int s=0; s < MCsamples; s++)  {
           covbranchalpha += (GetMCfbranchalpha(s) - meanfbranchalpha) * (GetMChbranchalpha(s) - meanhbranchalpha);
           varbranchalpha += (GetMChbranchalpha(s) - meanhbranchalpha) * (GetMChbranchalpha(s) - meanhbranchalpha);    
   }
   covbranchalpha /= MCsamples;
   varbranchalpha /= MCsamples;    
   a_branchalpha = covbranchalpha / varbranchalpha;
 
   for (int s=0; s < MCsamples; s++) {
            grad_branchalpha += (GetMCfbranchalpha(s) - a_branchalpha * GetMChbranchalpha(s));          
   } 
   grad_branchalpha /= MCsamples;
   
   delete[] MCDQbranchalpha;
   delete[] MCLogLikelihoodBranch;
   delete[] MCLogQBranchLengthPrior;
   delete[] MCLogBranchLengthPrior;
   delete[] MCfbranchalpha;
   delete[] MChbranchalpha;
} 

// ------------------------------------------------------------------------------------------------------------------------------------------

void GammaBranchProcessVI::MCQbranchbeta(int MCsamples) {
   
   MCLogLikelihoodBranch = new double[MCsamples];
   MCLogQBranchLengthPrior = new double[MCsamples];
   MCLogBranchLengthPrior = new double[MCsamples];
   MCfbranchbeta = new double[MCsamples];
   MChbranchbeta = new double[MCsamples];
   MCDQbranchbeta = new double[MCsamples];

   GlobalUpdateBranchLengthSuffStat();
   
   for (int s=0; s < MCsamples; s++) {
        MCDQbranchbeta[s] = 0;
        MCLogLikelihoodBranch[s] = 0;
        MCLogQBranchLengthPrior[s] = 0;
        MCLogBranchLengthPrior[s] = 0;
        MCfbranchbeta[s] = 0;
        MChbranchbeta[s] = 0; 
        MCblarray = new double[GetNbranch()]; 
             for (int i=1; i<GetNbranch(); i++)	{
             MCblarray[i] = rnd::GetRandom().Gamma(branchalpha,branchbeta);
             MCDQbranchbeta[s] += branchalpha/branchbeta - MCblarray[i];
             MCLogLikelihoodBranch[s] -= MCblarray[i] * GetBranchLengthSuffStatBeta(i); 
             MCLogLikelihoodBranch[s] += log(MCblarray[i]) * GetBranchLengthSuffStatCount(i);
             MCLogBranchLengthPrior[s] += Pbranchalpha * log(Pbranchbeta) - rnd::GetRandom().logGamma(Pbranchalpha) + (Pbranchalpha-1) * log(MCblarray[i]) - Pbranchbeta * MCblarray[i];
             MCLogQBranchLengthPrior[s] += branchalpha * log(branchbeta) - rnd::GetRandom().logGamma(branchalpha) + (branchalpha -1) * log(MCblarray[i]) - branchbeta * MCblarray[i];
             }
        delete[] MCblarray;
        MCfbranchbeta[s] = MCDQbranchbeta[s] * (MCLogBranchLengthPrior[s] - MCLogQBranchLengthPrior[s] + MCLogLikelihoodBranch[s]);
        MChbranchbeta[s] = MCDQbranchbeta[s];
    }
}

/*void GammaBranchProcessVI::GlobalMCQbranchbeta(int MCsamples) {
     assert(GetMyid() == 0);

     MPI_Status stat;
     MESSAGE signal = MC_BBETA;
     MPI_Bcast(&signal,1,MPI_INT,0,MPI_COMM_WORLD);
     int itmp[1];
     itmp[0] = MCsamples;
     MPI_Bcast(itmp,1,MPI_INT,0,MPI_COMM_WORLD); 
     
     int nprocs = GetNprocs();
     double ivector[MCsamples];
     double dvector[MCsamples];
        for(int i=1; i<nprocs; ++i) {
              MPI_Recv(ivector,MCsamples,MPI_DOUBLE,MPI_ANY_SOURCE,TAG1,MPI_COMM_WORLD,&stat);
              for(int j=0; j < MCsamples; ++j) {
                  MCfbranchbeta[j] = ivector[j];
              }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i=1; i<nprocs; ++i)  {
              MPI_Recv(dvector,MCsamples,MPI_DOUBLE,MPI_ANY_SOURCE,TAG1,MPI_COMM_WORLD,&stat);
              for(int j=0; j < MCsamples; ++j)  {
                  MChbranchbeta[j] = dvector[j];
              }
        }
}
 
void GammaBranchProcessVI::SlaveMCQbranchbeta()  {
   assert(GetMyid() > 0);
   
   int itmp[1];
   MPI_Bcast(itmp,1,MPI_INT,0,MPI_COMM_WORLD);
   int MCsamples = itmp[0];
   
   MCLogLikelihoodBranch = new double[MCsamples];
   MCLogQBranchLengthPrior = new double[MCsamples];
   MCLogBranchLengthPrior = new double[MCsamples];
   MCfbranchbeta = new double[MCsamples];
   MChbranchbeta = new double[MCsamples];
   MCDQbranchbeta = new double[MCsamples];

   GlobalUpdateBranchLengthSuffStat();
   
   for (int s=0; s < MCsamples; s++) {
        MCDQbranchbeta[s] = 0;
        MCLogLikelihoodBranch[s] = 0;
        MCLogQBranchLengthPrior[s] = 0;
        MCLogBranchLengthPrior[s] = 0;
        MCfbranchbeta[s] = 0;
        MChbranchbeta[s] = 0; 
        MCblarray = new double[GetNbranch()]; 
             for (int i=1; i<GetNbranch(); i++)	{
             MCblarray[i] = rnd::GetRandom().Gamma(branchalpha,branchbeta);
             MCDQbranchbeta[s] += branchalpha/branchbeta - MCblarray[i];
             MCLogLikelihoodBranch[s] -= MCblarray[i] * GetBranchLengthSuffStatBeta(i); 
             MCLogLikelihoodBranch[s] += log(MCblarray[i]) * GetBranchLengthSuffStatCount(i);
             MCLogBranchLengthPrior[s] += Pbranchalpha * log(Pbranchbeta) - rnd::GetRandom().logGamma(Pbranchalpha) + (Pbranchalpha-1) * log(MCblarray[i]) - Pbranchbeta * MCblarray[i];
             MCLogQBranchLengthPrior[s] += branchalpha * log(branchbeta) - rnd::GetRandom().logGamma(branchalpha) + (branchalpha -1) * log(MCblarray[i]) - branchbeta * MCblarray[i];
             }
        delete[] MCblarray;
        MCfbranchbeta[s] = MCDQbranchbeta[s] * (MCLogBranchLengthPrior[s] - MCLogQBranchLengthPrior[s] + MCLogLikelihoodBranch[s]);
        MChbranchbeta[s] = MCDQbranchbeta[s];
    }
  MPI_Send(MCfbranchbeta,MCsamples,MPI_INT,0,TAG1,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);   
  MPI_Send(MChbranchbeta,MCsamples,MPI_INT,0,TAG1,MPI_COMM_WORLD);
}
*/
void GammaBranchProcessVI::GradQbranchbeta(int MCsamples)   {

    /* GlobalMCQbranchbeta(MCsamples); */

    MCQbranchbeta(MCsamples);
    grad_branchbeta = 0;
    meanfbranchbeta = 0;
    meanhbranchbeta = 0; 
      
    for (int s=0; s < MCsamples; s++) {
            meanfbranchbeta += GetMCfbranchbeta(s);
            meanhbranchbeta += GetMChbranchbeta(s); 
    }
    meanfbranchbeta /= MCsamples;
    meanhbranchbeta /= MCsamples;
    varbranchbeta = 0;
    covbranchbeta = 0;

    for (int s=0; s < MCsamples; s++)  {
           covbranchbeta += (GetMCfbranchbeta(s) - meanfbranchbeta) * (GetMChbranchbeta(s) - meanhbranchbeta);
           varbranchbeta += (GetMChbranchbeta(s) - meanhbranchbeta) * (GetMChbranchbeta(s) - meanhbranchbeta);    
    }
    covbranchbeta /= MCsamples;
    varbranchbeta /= MCsamples;
    a_branchbeta = covbranchbeta / varbranchbeta;
 
    for (int s=0; s < MCsamples; s++) {
            grad_branchbeta += (GetMCfbranchbeta(s) - a_branchbeta * GetMChbranchbeta(s));            
    }
    grad_branchbeta /= MCsamples; 
  
  delete[] MCLogLikelihoodBranch;
  delete[] MCLogQBranchLengthPrior;
  delete[] MCLogBranchLengthPrior;
  delete[] MCfbranchbeta;
  delete[] MChbranchbeta;
  delete[] MCDQbranchbeta;
  
}

//---------------------------------------------------------------------------------------------------------------------------
//                             Stochastic Optimization for Variational Parameters  
//---------------------------------------------------------------------------------------------------------------------------
 
double GammaBranchProcessVI::Movebranchalpha(int MCsamples)  {
     GradQbranchalpha(MCsamples);
     Gbranchalpha += Getgrad_branchalpha() * Getgrad_branchalpha(); 
     branchalpha += (0.01 * Getgrad_branchalpha()) / sqrt(Gbranchalpha + 1e-8);
   return branchalpha;  
}

double GammaBranchProcessVI::Movebranchbeta(int MCsamples)  {
       GradQbranchbeta(MCsamples);
       Gbranchbeta += Getgrad_branchbeta() * Getgrad_branchbeta();
       branchbeta += (0.01 * Getgrad_branchbeta()) / sqrt(Gbranchbeta + 1e-8); 
   return branchbeta;
}

double GammaBranchProcessVI::Move(int MCsamples)  {
       Movebranchalpha(MCsamples);
       Movebranchbeta(MCsamples);
       MoveLength();
       return 1.0;
}

