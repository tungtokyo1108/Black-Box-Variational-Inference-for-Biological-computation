
/********************


**********************/


#include "PoissonSBDPProfileProcessVI.h"
#include <cassert>
#include "Parallel.h"

double PoissonSBDPProfileProcessVI::GlobalMixMove(int MCsamples, int nallocrep, double epsilon)	{

	if (Ncomponent != GetNmodeMax())	{
		cerr << "error in sbdp inc dp move: number of components\n";
		exit(1);
	}

	int K0 = GetNmodeMax();
	if (epsilon)	{
		double r = kappa / (1 + kappa);
		K0 = (int) (log(epsilon) / log(r));
		if (K0 >= GetNmodeMax())	{
			K0 = GetNmodeMax();
		}
	}

	// send mixmove signal and tuning parameters
	MESSAGE signal = MIX_MOVE;
	MPI_Bcast(&signal,1,MPI_INT,0,MPI_COMM_WORLD);
	int itmp[3];
	itmp[0] = MCsamples;
	itmp[1] = nallocrep;
	itmp[2] = K0;
	MPI_Bcast(itmp,3,MPI_INT,0,MPI_COMM_WORLD);

	// split Nsite among GetNprocs()-1 slaves
	int width = GetNsite()/(GetNprocs()-1);
	int smin[GetNprocs()-1];
	int smax[GetNprocs()-1];
	for(int i=0; i<GetNprocs()-1; ++i) {
		smin[i] = width*i;
		smax[i] = width*(1+i);
		if (i == (GetNprocs()-2)) smax[i] = GetNsite();
	}

	double* tmp = new double[Ncomponent * GetDim() + 1];	

	MPI_Bcast(weight,Ncomponent,MPI_DOUBLE,0,MPI_COMM_WORLD);

		// here slaves do realloc moves

		// mpi receive new allocations
		MPI_Status stat;
		int tmpalloc[GetNsite()+1];
		for(int i=1; i<GetNprocs(); ++i) {
			MPI_Recv(tmpalloc,GetNsite(),MPI_INT,i,TAG1,MPI_COMM_WORLD,&stat);
			for(int j=smin[i-1]; j<smax[i-1]; ++j) {
				alloc[j] = tmpalloc[j];
				if ((alloc[j] < 0) || (alloc[j] >= Ncomponent))	{
					cerr << "alloc overflow\n";
					exit(1);
				}
			}
		}

		// MPI_Barrier(MPI_COMM_WORLD);

		// broadcast new allocations
		MPI_Bcast(alloc,GetNsite(),MPI_INT,0,MPI_COMM_WORLD);

		// here slaves do profile moves

		UpdateOccupancyNumbers();

		// collect final values of profiles (+ total acceptance rate) from slaves

		// split Nmode among GetNprocs()-1 slaves
		int mwidth = GetNcomponent()/(GetNprocs()-1);
		int mmin[GetNprocs()-1];
		int mmax[GetNprocs()-1];
		for(int i=0; i<GetNprocs()-1; ++i) {
			mmin[i] = mwidth*i;
			mmax[i] = mwidth*(1+i);
			if (i == (GetNprocs()-2)) mmax[i] = GetNcomponent();
		}

		MPI_Status stat2;
		double total = 0;
		for(int i=1; i<GetNprocs(); ++i) {
			MPI_Recv(tmp,(mmax[i-1]-mmin[i-1])*GetDim()+1,MPI_DOUBLE,i,TAG1,MPI_COMM_WORLD,&stat2);
			int l = 0;
			for(int j=mmin[i-1]; j<mmax[i-1]; ++j) {
				for (int k=0; k<GetDim(); k++)	{
					profile[j][k] = tmp[l];
					l++;
				}
			}
			total += tmp[l]; // (sum all acceptance rates)
		}

		// MPI_Barrier(MPI_COMM_WORLD);
		// resend all profiles
		MPI_Bcast(allocprofile,Ncomponent*GetDim(),MPI_DOUBLE,0,MPI_COMM_WORLD);

	delete[] tmp;

	return 1;
}


void PoissonSBDPProfileProcessVI::SlaveMixMove()	{

	int itmp[3];
	MPI_Bcast(itmp,3,MPI_INT,0,MPI_COMM_WORLD);
	int MCsamples = itmp[0];
	int nallocrep = itmp[1];
        int K0 = itmp[2];

	double* mLogSamplingArray = new double[Ncomponent];
	double* tmp = new double[Ncomponent * GetDim() + 1];
	int width = GetNsite()/(GetNprocs()-1);
	int smin[GetNprocs()-1];
	int smax[GetNprocs()-1];
	for(int i=0; i<GetNprocs()-1; ++i) {
		smin[i] = width*i;
		smax[i] = width*(1+i);
		if (i == (GetNprocs()-2)) smax[i] = GetNsite();
	}

		// realloc move
        MPI_Bcast(weight,Ncomponent,MPI_DOUBLE,0,MPI_COMM_WORLD);
		// MPI_Bcast(allocprofile,Ncomponent*GetDim(),MPI_DOUBLE,0,MPI_COMM_WORLD);

        double totp = 0;
	for (int mode = 0; mode<K0; mode++)   {
	         totp += weight[mode];
		}
	double totq = 0;
	for (int mode=K0; mode<GetNmodeMax(); mode++)	{
			totq += weight[mode];
		}

	for (int allocrep=0; allocrep<nallocrep; allocrep++)	{

	    for (int site=smin[GetMyid()-1]; site<smax[GetMyid()-1]; site++)	{
	      // for (int site=GetSiteMin(); site<GetSiteMax(); site++)	{

                   /*Gpcat = new double[K0];
                   PCAT = new double[K0];
                   for (int mode = 0; mode < K0; mode++) {
                     Gpcat[mode] = 0;
                     PCAT[mode] = 0;
                   }*/                   

                   double max =0;
		   for (int mode = 0; mode<K0; mode++)	{
                        mLogSamplingArray[mode] = LogStatProb(site,mode);
                        if ((!mode) || (max < mLogSamplingArray[mode])) {
                              max = mLogSamplingArray[mode];
                        }
                   }
                   
                   meanfpcat = new double[K0];
                   meanhpcat = new double[K0];
                   varpcat = new double[K0];
                   covpcat = new double[K0];
                   a_pcat = new double[K0];
                   grad_pcat = new double[K0];
                   for (int mode = 0; mode < K0; mode++)  {
                        MCweight = new double[MCsamples];
                        MCdpcat = new double[MCsamples];
                        MCLogSamplingArray = new double[MCsamples];
                        MCLogQpcat = new double[MCsamples];
                        MCLogpcat = new double[MCsamples];
                        MCfpcat = new double[MCsamples];
                        MChpcat = new double[MCsamples];
          
                        meanfpcat[mode] = 0;
                        meanhpcat[mode] = 0;
                        varpcat[mode] = 0;
                        covpcat[mode] = 0;
                        a_pcat[mode] = 0;
                        grad_pcat[mode] = 0;
                        for (int s=0; s<MCsamples; s++)	{
                                ResampleWeights();
                                MCweight[s] = weight[mode];
                                MCdpcat[s] =  MCweight[s] * (1/PCAT[mode]);
                                MCLogSamplingArray[s] = MCweight[s] * (mLogSamplingArray[mode] - max);
                                MCLogQpcat[s] = MCweight[s] * log(PCAT[mode]);
                                MCLogpcat[s] = totq * log(1 - V[mode]) + MCweight[s] * log(V[mode]);                                 
				MCfpcat[s] = MCdpcat[s] * (MCLogpcat[s] - MCLogQpcat[s] + MCLogSamplingArray[s]);
                                MChpcat[s] = MCdpcat[s];
                                meanfpcat[mode] += MCfpcat[s];
                                meanhpcat[mode] += MChpcat[s];
                        }
                        delete[] MCweight;
                        delete[] MCdpcat;
                        delete[] MCLogSamplingArray;
                        delete[] MCLogQpcat;
                        delete[] MCLogpcat;

                        meanfpcat[mode] /= MCsamples;
                        meanhpcat[mode] /= MCsamples;
                        
                        for (int s = 0; s < MCsamples; s++)  {
                              covpcat[mode] += (MCfpcat[s] - meanfpcat[mode]) * (MChpcat[s] - meanhpcat[mode]);
                              varpcat[mode] += (MChpcat[s] - meanhpcat[mode]) * (MChpcat[s] - meanhpcat[mode]);
                        }
                        covpcat[mode] /= MCsamples;
                        varpcat[mode] /= MCsamples;                        
                        a_pcat[mode] = covpcat[mode] / varpcat[mode];

                        for (int s = 0; s < MCsamples; s++)  {
                              grad_pcat[mode] += (MCfpcat[s] - a_pcat[mode] * MChpcat[s]);
                        }
                        grad_pcat[mode] /= MCsamples;

                        Gpcat[mode] += grad_pcat[mode] * grad_pcat[mode];
                        PCAT[mode] += (0.01 * grad_pcat[mode]) / sqrt(Gpcat[mode] + 1e-8);
                  }
                  delete[] meanfpcat;
                  delete[] meanhpcat;
                  delete[] varpcat;
                  delete[] covpcat;
                  delete[] a_pcat;
                  delete[] grad_pcat;

                  maxPCAT = 0;
                  for (int mode = 0; mode < K0; mode++) {
                        if ((!mode) || (maxPCAT < PCAT[mode]))  {
                               maxPCAT = PCAT[mode];
                               alloc[site] = mode;
                        }
                  }
                            
           }
        }
		MPI_Send(alloc,GetNsite(),MPI_INT,0,TAG1,MPI_COMM_WORLD);

		// MPI_Barrier(MPI_COMM_WORLD);
		// profile move

		// receive new allocations
		MPI_Bcast(alloc,GetNsite(),MPI_INT,0,MPI_COMM_WORLD);

		// determine the range of components to move
		UpdateOccupancyNumbers();

		// update sufficient statistics
		UpdateModeProfileSuffStat();

		// split Nmode among GetNprocs()-1 slaves
		int mwidth = GetNcomponent()/(GetNprocs()-1);
		int mmin[GetNprocs()-1];
		int mmax[GetNprocs()-1];
		for(int i=0; i<GetNprocs()-1; ++i) {
			mmin[i] = mwidth*i;
			mmax[i] = mwidth*(1+i);
			if (i == (GetNprocs()-2)) mmax[i] = GetNcomponent();
		}

		double total = 0;
		for (int mode=mmin[GetMyid()-1]; mode<mmax[GetMyid()-1]; mode++)	{
			total += MoveProfile(mode);
		}
		int l = 0;
		for (int mode=mmin[GetMyid()-1]; mode<mmax[GetMyid()-1]; mode++)	{
			for (int k=0; k<GetDim(); k++)	{
				tmp[l] = profile[mode][k];
				l++;
			}
		}
		tmp[l] = total;

		MPI_Send(tmp,(mmax[GetMyid()-1] - mmin[GetMyid()-1])*GetDim()+1,MPI_DOUBLE,0,TAG1,MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);

		// rereceive all profiles
		MPI_Bcast(allocprofile,Ncomponent*GetDim(),MPI_DOUBLE,0,MPI_COMM_WORLD);

	delete[] mLogSamplingArray;
	delete[] tmp;
}

