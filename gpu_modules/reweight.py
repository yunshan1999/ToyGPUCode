pandax4t_reweight_fitting = """
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
extern "C" {

  __device__ float interpolate1d(float x, float * array_x, float * array_y){
    int ndimx = (int)array_x[0];
    int ndimy = (int)array_y[0];
    if(ndimx != ndimy){
      printf("two arrays not in same dimension!");
      return 0.;
    }
    else{
      if(x < array_x[1])return array_y[1];
      else if(x>array_x[ndimx])return array_y[ndimx];
      else{
        int i;
        for(i = 1; i <= ndimx ; i++){
          if(x > array_x[i]);
          else break;
        }
        float xmin = array_x[i-1];
        float xmax = array_x[i];
        float ymin = array_y[i-1];
        float ymax = array_y[i];
        float temp = ymin + (ymax - ymin)/(xmax - xmin) * (x - xmin);
        return temp;
      }

    }
  }

  __device__ float get_tritium_energy_weight(float energy,float *par)
  {
    float flat=par[0];
    float T = energy;
    float Q(18.5906),pi(3.1415926);
    if (T>Q||T<0)  return flat;
    float m = 0.511e3;
    float P = sqrt(2*m*T);
    float E = T+m;
    float eta = 2.*1./137*E/P;
    float F = 2*pi*eta/(1-exp(-2*pi*eta));
    return float(F*P*E*(Q-T)*(Q-T)*0.00001+flat);

  }

  __device__ float get_s1_efficiency(int nHitsS1){
    float hit[] = {20,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    float eff[] = {20,
      0.,0.,0.3442748091603054,0.6625954198473284,0.7839694656488551,
      0.836641221374046,0.8412213740458017,0.8893129770992368,
      0.9030534351145039,0.9053435114503818,0.919083969465649,
      0.9328244274809161,0.9442748091603055,0.9511450381679392,
      0.9534351145038169,0.9534351145038169,0.9603053435114505,
      0.9648854961832063,0.9717557251908397,0.9763358778625955
    };
    float Eff = interpolate1d((float)nHitsS1, hit, eff);
    return Eff;
  }


  __device__ float get_g1_true(float dt, float g1mean){
    float A = 5.64008e2;
    float B = 3.92408e-1;
    float C = -1.65968e-4;
    float D = 6.69171e-7;
    float qS1maxmean = 803.319;
    float qS1max = A + B * dt + C * dt * dt + D * dt * dt * dt;
    float g1_true = g1mean * qS1max/qS1maxmean;
    return g1_true;
  }


  __device__ float truncated_normal_ratio(float obs, float mean1, float sigma1, float mean2, float sigma2, float lower, float upper)
  {
    float R = 1.;
    if (obs<lower || obs>upper) return 0;
    // need to calculate the scale factors for trucated gaussian
    float scale_factor1 = 0.5*(erff( (upper-mean1)/sigma1 / sqrtf(2.)) - erff( (lower-mean1)/sigma1/sqrtf(2.) ));
    float scale_factor2 = 0.5*(erff( (upper-mean2)/sigma2 / sqrtf(2.)) - erff( (lower-mean2)/sigma2/sqrtf(2.) ));
    R *= scale_factor2/scale_factor1 * sigma2 / sigma1;//denominator terms
    R *= expf( - 0.5*powf((obs - mean1)/sigma1, 2.) + 0.5*powf( (obs-mean2)/sigma2, 2.) );//numerator terms
    return R;
  }


  __device__ float normal_ratio(float obs, float mean1, float sigma1, float mean2, float sigma2)
  {
    double R = 1.;
    R *= double(sigma2) / double(sigma1);
    R *= exp( - 0.5*pow((double(obs) - double(mean1))/double(sigma1), 2.) + 0.5*pow( (double(obs)-double(mean2))/double(sigma2), 2.) );
    return float(R);
  }


//  __device__ float binomial_ratio(int obs, int N, float prob1, float prob2)
//{
//    float R = 1.;
//    R *= powf(prob1/prob2, (float)obs);
//    R *= powf((1.-prob1)/(1.-prob2), (float)(N - obs));
//    return R;
//}
  
  __device__ float binomial_ratio(int obs, int N, float prob1, float prob2)
  {
    if( float(prob1) == float(prob2) && ( prob2 == float(0.) || prob2 == float(1.)) ) return 1.;
    else if(prob2 == float(1.) || prob2 == float(0.) || prob1 == 1. || prob1 == 0.) return 0.;

    if(float(N) * prob2 > 20 && float(N) * (1. - prob2) > 20){
      float sigma_new = sqrtf(float(N) * prob1 * (1. - prob1));
      float mean_new  = float(N) * prob1;
      float sigma = sqrtf(float(N) * prob2 * (1. - prob2));
      float mean = float(N) * prob2;
      return normal_ratio(float(obs), mean_new,sigma_new,mean,sigma);
    
    }

    float R = 1.;
    R *= pow(double(prob1)/double(prob2), (double)obs);
    R *= pow((1.-double(prob1))/(1.-double(prob2)), (double)(N - obs));
    //if( float(prob1) = float(prob2) ) R = double(1.);
    return R;
  }


  __global__ void ln_likelihood(
      float* xdata, float* ydata, 
      double *histogram2d, double *histogram2d_par,
      double *lnlikelihood, int* Ndata)
  {
    int eventId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; //start with 1 
    //printf("test i: %d %d",int(*Ndata),eventId);

    //if(eventId > *(Ndata) ) return;
    float s1 = (float) *(xdata+eventId);
    float s2 = (float) log10(double(*(ydata+eventId)));


    double Nmc = (double) *(histogram2d_par);
    int xBinning = (int)*(histogram2d_par+1);
    float xMin = (float)*(histogram2d_par+2);
    float xMax = (float)*(histogram2d_par+3);
    float xStep = (double(xMax) - double(xMin))/double(xBinning);
    int yBinning = (int)*(histogram2d_par+4);
    float yMin = (float)*(histogram2d_par+5);
    float yMax = (float)*(histogram2d_par+6);
    float yStep = (double(yMax) - double(yMin))/double(yBinning);

    int xbin = (int) floor((double(s1) - double(xMin))/double(xStep)) ;
    int ybin = (int) floor((double(s2) - double(yMin))/double(yStep)) ;
    double bincontent = *((double*) ((char*)histogram2d + ybin*xBinning * sizeof(double)) + xbin);
    //if(eventId <20) printf("test s1, log10(s2),bincontent: %.2f, %.2f, %.2f\\n",s1,s2,bincontent);
    //double bincontent = *data_address;
    if (bincontent == 0.0) atomicAdd(&lnlikelihood[0],-100.0);
    else if (bincontent > 0.0) atomicAdd(&lnlikelihood[0],(log10(bincontent/Nmc)));


  }



  __device__ void get_yield_pars(bool simuTypeNR, float E_drift, float energy, float density, float * pars, float * free_pars){
    //This function calculates Lindhaed, W, Nex/Ni, recomb fraction mean and delta and returns them all
    //ATTENTION: Just to clarify.NuisParam&FreeParam are NESTv2 Pars, not the ones we are going to fit ni this model. Here we just regard them as intrinsic fixed pars.

    float nest_avo = 6.0221409e+23;
    float molar_mass = 131.293;
    float atom_num = 54.;
    float eDensity = (density/molar_mass) * nest_avo * atom_num;
    float Wq_eV = 18.7263 - 1.01e-23 * eDensity;
    Wq_eV *= 1.1716263232;
    if(simuTypeNR){
      float NuisParam[11] = {11.,1.1,0.0480,-0.0533,12.6,0.3,2.,0.3,2.,0.5,1.};
      float Nq = NuisParam[0] * powf(energy, NuisParam[1]);
      float ThomasImel =
        NuisParam[2] * powf(E_drift, NuisParam[3]) * powf(density / 2.90, 0.3);
      float Qy = 1. / (ThomasImel*powf(energy+NuisParam[4],NuisParam[9]));
      Qy *= 1. - 1. / powf(1. + powf((energy / NuisParam[5]), NuisParam[6]),NuisParam[10]);
      float Ly = Nq / energy - Qy;
      if (Qy < 0.0) Qy = 0.0;
      if (Ly < 0.0) Ly = 0.0;
      float Ne = Qy * energy;
      float Nph = Ly * energy *
        (1. - 1. / (1. + powf((energy / NuisParam[7]), NuisParam[8])));
      Nq = Nph + Ne;
      float Ni = (4. / ThomasImel) * (expf(Ne * ThomasImel / 4.) - 1.);
      float Nex = (-1. / ThomasImel) * (4. * expf(Ne * ThomasImel / 4.) -
          (Ne + Nph) * ThomasImel - 4.);
      float elecFrac = Ne/(Ne+Nph);
      float NexONi = Nex / Ni;
      float rmean = 1. - Ne / Ni;
      float FreeParam[] = {1.,1.,0.1,0.5,0.19,2.25};
      float omega = FreeParam[2] * expf(-0.5 * powf(elecFrac - FreeParam[3],2.)/(FreeParam[4] * FreeParam[4]));
      if(omega < 0.)omega = 0.;
      double L = (Nq / energy) * Wq_eV * 1e-3;
      pars[0] = Wq_eV;
      pars[1] = L;
      pars[2] = NexONi;
      pars[3] = rmean * (1 + free_pars[4] + free_pars[5]*energy);
      pars[4] = omega * free_pars[6];
      if(pars[3]<0.)pars[3] = 0.;
      if(pars[3]>0.9)pars[3] = 0.9;
      if(pars[4]<0.001)pars[4] = 0.001;
      return;
    }
    else{
      // float Wq_eV = 1.9896 + (20.8 - 1.9896) / (1. + powf(density / 4.0434, 1.4407));

      float QyLvllowE = 1e3 / Wq_eV + 6.5 * (1. - 1. / (1. + powf(E_drift / 47.408, 1.9851)));
      float HiFieldQy =
        1. + 0.4607 / powf(1. + powf(E_drift / 621.74, -2.2717), 53.502);
      float QyLvlmedE =
        32.988 -
        32.988 /
        (1. + powf(E_drift / (0.026715 * expf(density / 0.33926)), 0.6705));
      QyLvlmedE *= HiFieldQy;
      float DokeBirks =
        1652.264 +
        (1.415935e10 - 1652.264) / (1. + powf(E_drift / 0.02673144, 1.564691));
      float Nq = energy * 1e3 /
        Wq_eV;  //( Wq_eV+(12.578-Wq_eV)/(1.+powf(energy/1.6,3.5)) );
      float LET_power = -2.;
      float QyLvlhighE = 28.;
      //      if (density > 3.) QyLvlhighE = 49.; Solid Xe effect from Yoo. But,
      //      beware of enabling this line: enriched liquid Xe for neutrinoless
      //      double beta decay has density higher than 3g/cc;
      float Qy = QyLvlmedE +
        (QyLvllowE - QyLvlmedE) /
        powf(1. + 1.304 * powf(energy, 2.1393), 0.35535) +
        QyLvlhighE / (1. + DokeBirks * powf(energy, LET_power));
      if (Qy > QyLvllowE && energy > 1. && E_drift > 1e4) Qy = QyLvllowE;
      float Ly = Nq / energy - Qy;
      float Ne = Qy * energy;
      float Nph = Ly * energy;
      float alpha = 0.067366 + density * 0.039693;
      float NexONi = alpha * erff(0.05 * energy);
      float Nex = Nq * (NexONi) / (1. + NexONi);
      float Ni = Nq * 1. / (1. + NexONi);
      float rmean = 1 - Ne / Ni;
      float elecFrac = Ne/(Ne+Nph);
      float ampl = 0.14 + (0.043 - 0.14)/(1. + powf(E_drift/1210.,1.25)); //0.086036+(0.0553-0.086036)/powf(1.+powf(E_drift/295.2,251.6),0.0069114); //pair with GregR mean yields model
      if(ampl < 0.)ampl = 0.;
      float wide = 0.205; //or: FreeParam #2, like amplitude (#1)
      float cntr = 0.5; //0.41-45 agrees better with Dahl thesis. Odd! Reduces fluctuations for high e-Frac (high EF,low E). Also works with GregR LUX Run04 model. FreeParam #3
      //for gamma-rays larger than 100 keV at least in XENON10 use 0.43 as the best fit. 0.62-0.37 for LUX Run03
      float skew = -0.2; //FreeParam #4
      float mode = cntr + sqrtf(2./M_PI)*skew*wide/sqrtf(1.+skew*skew);
      float norm = 1./(expf(-0.5*powf(mode-cntr,2.)/(wide*wide))*(1.+erff(skew*(mode-cntr)/(wide*sqrtf(2.))))); //makes sure omega never exceeds ampl
      float omega = norm*ampl*expf(-0.5*powf(elecFrac-cntr,2.)/(wide*wide))*(1.+erff(skew*(elecFrac-cntr)/(wide*sqrtf(2.))));
      if(omega<0.)omega = 0;
      pars[0] = Wq_eV;
      pars[1] = 1.;
      pars[2] = NexONi;
      pars[3] = rmean * (1 + free_pars[4] + free_pars[5]*energy);
      pars[4] = omega * free_pars[6];
      if(pars[3]<0.)pars[3] = 0.;
      if(pars[3]>0.9)pars[3] = 0.9;
      if(pars[4]<0.)pars[4] = 0.;
      return;
    }
  }


  __global__ void main_reweight_fitting(float *new_v_variable,
      float *inTree, float *inTree_par,
      double *output, double *output_par, int mode)
  {
    // initiate the random sampler, must have.
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    // get the number of trials
    int num_trials      = (int) *(inTree_par);
    if (iteration>num_trials|| iteration<1) return;
    int stride_inTree = (int) *(inTree_par+1);
    float* entry_data = (float*)((char*)inTree + iteration * stride_inTree);

    float pulseAreaS1Cor = (float) *(entry_data);
    float pulseAreaS2Cor = (float) *(entry_data+1);
    float dt = (float) *(entry_data+2);
    float eLife_us = (float) *(entry_data+3);

    float g1 = (float) *(entry_data+4);
    float Nph = (float) *(entry_data+5);
    float nHitsS1 = (float) *(entry_data+6);

    float eee = (float) *(entry_data+7);
    float Ne = (float) *(entry_data+8);
    float Nee = (float) *(entry_data+9);
    float weight = (float) *(entry_data+10);
    float seg = (float) *(entry_data+11);
    float nHitsS2 = (float) *(entry_data+12);
    float bottom_fraction = (float) *(entry_data+13);
    float pulseAreaS2CorB = (float) *(entry_data+14);
    float w_eV = *(entry_data+15);
    float L = (float) *(entry_data+16);
    float NexONi= (float) *(entry_data+17);
    float rmean = (float) *(entry_data+18);
    float deltaR = (float) *(entry_data+19);
    float energy = (float) *(entry_data+20);
    float Nq_actual = (float) *(entry_data+21);
    float Nq = (float) *(entry_data+22);
    float Ni = (float) *(entry_data+23);
    float r = (float) *(entry_data+24);
    float pulseAreaBiasS2 = (float) *(entry_data+25);


    //the real random walk space is half of the initial data for every dimension
    float g1_new =  (float) *(new_v_variable);
    float eee_new =  (float) *(new_v_variable+1);
    float seg_new =  (float) *(new_v_variable+2);
    float bottom_fraction_new =  (float) *(new_v_variable+3);
    float p0Recomb_new = (float) *(new_v_variable+4);
    float p1Recomb_new = (float) *(new_v_variable+5);
    float p0FlucRecomb_new = (float) *(new_v_variable+6);
    float flatE_new = (float) *(new_v_variable+7);
    float par_e[]={flatE_new};
    float energy_weight_new = get_tritium_energy_weight(energy,par_e);
    
    float pars[5]={0};

    float E_drift = 115.6; //drift electric field in V/cm
    float density = 2.8611; //density of liquid xenon
    bool simuTypeNR = false;

    get_yield_pars(simuTypeNR, E_drift, energy, density, &pars[0], new_v_variable);
    float w_eV_new = pars[0];
    float L_new = pars[1];//lindhard factor
    float NexONi_new = pars[2];
    float rmean_new = pars[3];
    float deltaR_new = pars[4];

    float g1_true = get_g1_true(dt,g1);
    float g1_true_new = get_g1_true(dt,g1_new);

    double ReweightingWeight = 1.;
    ReweightingWeight *= (double) binomial_ratio(Nq, Nq_actual, L_new,L);
    double r1 = (double) truncated_normal_ratio(r,rmean_new,deltaR_new,rmean,deltaR,0.,1.);
    double r2 = (double) binomial_ratio(int(pulseAreaS2CorB),int(pulseAreaS2Cor),bottom_fraction_new,bottom_fraction);
    double r3 = (double) binomial_ratio(int(nHitsS1), int(Nph), g1_true_new, g1_true);
    double r4 = (double) binomial_ratio(int(Nee), int(Ne), exp(-double(dt) / double(eLife_us)) * double(eee_new),exp(-double(dt) / double(eLife_us)) * double(eee));
    double r5 = (double) normal_ratio(nHitsS2, double(seg_new) * double(Nee), sqrt(double(seg_new) * double(Nee)), double(seg) * double(Nee), sqrt(double(seg) * double(Nee)));
    ReweightingWeight *= double(r1 * r2 * r3 * r4 * r5);

    //int offsetCheck =1000;
    //if(iteration >200 + offsetCheck && iteration <300 + offsetCheck)
    //{
    //  printf("i %d,  r1 %f,  r2 %f,  r3 %f, r4 %f, r5 %f\\n",iteration, r1 ,r2 , r3, r4, r5);
    //  double test = pow(5.3,1000);
    //  printf("i %d, %f, %f, %f, %f, %f, %f\\n",iteration, nHitsS1, Nph, g1_true_new, g1_true, r3, test);
    //}


    if (pulseAreaS2Cor < 80.  || pulseAreaS2CorB == 0. || pulseAreaS1Cor < 2 )ReweightingWeight = 0;
    
    for (int i=0;i<26;i++)
    {
      float enter_i = (float) *(entry_data+i);
      if( enter_i == float(0.))
      {
        ReweightingWeight = 0.;
        r1 = 0.;
        r2 = 0.;
        r3 = 0.;
        r4 = 0.;
        r5 = 0.;
        break;
      }
    }

    double new_weight = double(energy_weight_new) * ReweightingWeight;
    //filling new weight for mode = 1
    if(isnan(new_weight) ||isinf(new_weight)) 
    {
      new_weight = 0;

    }
    if(isnan(ReweightingWeight) ||isinf(ReweightingWeight)) 
    {
      ReweightingWeight = 0;

    }
    if(mode == 1 )
    {
      if(ReweightingWeight>10)printf("weird %d %f\\n",iteration,ReweightingWeight);

      if(iteration<100) printf("newweight %f \\n", new_weight);
      *(entry_data+10) =  (float) ReweightingWeight; 
      *(entry_data+15) =  (float) r1; 
      *(entry_data+16) =  (float) r2; 
      *(entry_data+17) =  (float) r3; 
      *(entry_data+21) =  (float) r4; 
      *(entry_data+22) =  (float) r5; 
      *(entry_data+23) =  (float) energy_weight_new; 
    }
    

    //filling ouput 2d histogram
    int xBinning = (int)*(output_par+1);
    double xMin = (double)*(output_par+2);
    double xMax = (double)*(output_par+3);
    double xStep = (xMax - xMin)/(double)xBinning;
    int yBinning = (int)*(output_par+4);
    double yMin = (double)*(output_par+5);
    double yMax = (double)*(output_par+6);
    double yStep = (yMax - yMin)/(double)yBinning;

    //get values, overflows and underflows in output[1]-output[9]
    double xvalue = (double)pulseAreaS1Cor;
    double yvalue = (double)log10f(pulseAreaS2CorB);

    //printf("I am block%d and thread%d for iteration%d \\n",blockIdx.x, threadIdx.x, iteration);
    //float* check_2d = (float*)((char*)output + 1 * xBinning * sizeof(float));
    //if (iteration < int(10) )
    //{
    //    printf("ori: g1, eee, seg, reweighting: %.2f, %.2f, %.2f, %.2f %f\\n",g1,eee,seg,ReweightingWeight, (float) *(check_2d));
    //}

    if(xvalue>=xMin && xvalue<xMax && yvalue>=yMin && yvalue<yMax)
    {
      int xbin = floorf((xvalue - xMin)/xStep) ;
      int ybin = floorf((yvalue - yMin)/yStep) ;
      double* data_address = (double*) ((char*)output + ybin*xBinning * sizeof(double)) + xbin;
      atomicAdd(data_address,new_weight);
      atomicAdd(output_par,new_weight);

    }



  }




}
"""
